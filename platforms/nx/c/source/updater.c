// In-app GitHub Releases auto-updater transport & atomic NRO swap helper.
// Uses Switch portlibs libcurl (backed by Horizon OS's native ssl service) so
// HTTPS and GitHub CDN redirects work cleanly without bundling extra TLS libs.

#include <curl/curl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <switch.h>

#define UNX_DEFAULT_NRO_PATH "sdmc:/switch/uprisingnx/uprising_nx.nro"

static char g_nro_path[512] = UNX_DEFAULT_NRO_PATH;
static atomic_bool g_pending_swap = false;
static pthread_once_t g_curl_once = PTHREAD_ONCE_INIT;

static void init_curl_once(void) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

static bool has_nro_suffix(const char *s) {
    size_t len = strlen(s);
    return len >= 4 && strcasecmp(s + len - 4, ".nro") == 0;
}

void unx_updater_set_nro_path(const char *argv0) {
    if (argv0 && strncmp(argv0, "sdmc:/", 6) == 0 && has_nro_suffix(argv0) && strlen(argv0) < sizeof(g_nro_path)) {
        snprintf(g_nro_path, sizeof(g_nro_path), "%s", argv0);
    } else {
        snprintf(g_nro_path, sizeof(g_nro_path), "%s", UNX_DEFAULT_NRO_PATH);
    }
    // Clean up any leftover .old backup from an interrupted previous swap.
    char old_path[528];
    snprintf(old_path, sizeof(old_path), "%s.old", g_nro_path);
    remove(old_path);
}

const char *unx_updater_nro_path(void) {
    return g_nro_path;
}

void unx_updater_mark_pending_swap(void) {
    atomic_store(&g_pending_swap, true);
}

// Fast check via nifm: returns 1 when the Switch has an active internet
// connection (Wi-Fi or Ethernet), 0 when offline / airplane mode.
int unx_updater_has_internet(void) {
    if (R_FAILED(nifmInitialize(NifmServiceType_User))) {
        return 0;
    }
    NifmInternetConnectionType conn_type = 0;
    u32 wifi_strength = 0;
    NifmInternetConnectionStatus conn_status = 0;
    Result rc = nifmGetInternetConnectionStatus(&conn_type, &wifi_strength, &conn_status);
    nifmExit();
    return (R_SUCCEEDED(rc) && conn_status == NifmInternetConnectionStatus_Connected) ? 1 : 0;
}

typedef struct {
    char *buf;
    size_t cap;
    size_t len;
} MemWriter;

static size_t write_to_mem(char *ptr, size_t size, size_t nmemb, void *userdata) {
    MemWriter *w = (MemWriter *)userdata;
    size_t n = size * nmemb;
    if (w->cap == 0) return 0;
    size_t avail = (w->cap - 1) - w->len;
    size_t take = n < avail ? n : avail;
    if (take > 0) {
        memcpy(w->buf + w->len, ptr, take);
        w->len += take;
        w->buf[w->len] = '\0';
    }
    // Returning n even if truncated prevents spurious write-error aborts when
    // GitHub returns long release notes; buffer is sized generously (128 KiB).
    return n;
}

static size_t write_to_file(char *ptr, size_t size, size_t nmemb, void *userdata) {
    return fwrite(ptr, size, nmemb, (FILE *)userdata) * size;
}

typedef struct {
    volatile int *progress_pct;
    const volatile int *cancel_flag;
} ProgressCtx;

static int on_xfer_progress(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    (void)ultotal;
    (void)ulnow;
    ProgressCtx *p = (ProgressCtx *)clientp;
    if (p->cancel_flag && *p->cancel_flag) {
        return 1; // abort transfer cleanly
    }
    if (p->progress_pct && dltotal > 0) {
        int pct = (int)((dlnow * 100) / dltotal);
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        *p->progress_pct = pct;
    }
    return 0;
}

static CURL *new_request(const char *url, struct curl_slist **headers_out) {
    pthread_once(&g_curl_once, init_curl_once);
    CURL *curl = curl_easy_init();
    if (!curl) return NULL;
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "UprisingNX-Switch-Updater");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 12L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    if (headers_out) {
        *headers_out = curl_slist_append(NULL, "Accept: application/vnd.github+json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, *headers_out);
    }
    return curl;
}

// Performs an HTTPS GET into `out_buf` (NUL-terminated). Returns 0 on success.
int unx_updater_http_get(
    const char *url,
    char *out_buf,
    size_t out_cap,
    size_t *out_len,
    long *http_status,
    const volatile int *cancel_flag,
    char *err_buf,
    size_t err_cap
) {
    if (out_len) *out_len = 0;
    if (http_status) *http_status = 0;
    if (out_buf && out_cap > 0) out_buf[0] = '\0';
    if (err_buf && err_cap > 0) err_buf[0] = '\0';

    struct curl_slist *headers = NULL;
    CURL *curl = new_request(url, &headers);
    if (!curl) {
        if (err_buf && err_cap > 0) snprintf(err_buf, err_cap, "curl_easy_init failed");
        return -1;
    }

    MemWriter writer = { out_buf, out_cap, 0 };
    ProgressCtx pctx = { NULL, cancel_flag };
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_mem);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &writer);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, on_xfer_progress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &pctx);

    CURLcode res = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    if (http_status) *http_status = status;
    if (out_len) *out_len = writer.len;

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "%s", curl_easy_strerror(res));
        }
        return (int)res;
    }
    return 0;
}

// Streams an HTTPS URL directly to `dest_path`, updating `*progress_pct` (0..100).
// Returns 0 on success.
int unx_updater_download_file(
    const char *url,
    const char *dest_path,
    volatile int *progress_pct,
    const volatile int *cancel_flag,
    char *err_buf,
    size_t err_cap
) {
    if (progress_pct) *progress_pct = 0;
    if (err_buf && err_cap > 0) err_buf[0] = '\0';

    FILE *f = fopen(dest_path, "wb");
    if (!f) {
        if (err_buf && err_cap > 0) snprintf(err_buf, err_cap, "Could not open %s for writing", dest_path);
        return -2;
    }

    CURL *curl = new_request(url, NULL);
    if (!curl) {
        fclose(f);
        remove(dest_path);
        if (err_buf && err_cap > 0) snprintf(err_buf, err_cap, "curl_easy_init failed");
        return -1;
    }

    ProgressCtx pctx = { progress_pct, cancel_flag };
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, on_xfer_progress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &pctx);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    bool closed_ok = (fclose(f) == 0);

    if (res != CURLE_OK || !closed_ok) {
        remove(dest_path);
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "%s", res != CURLE_OK ? curl_easy_strerror(res) : "SD write failed");
        }
        return res != CURLE_OK ? (int)res : -3;
    }
    if (progress_pct) *progress_pct = 100;
    return 0;
}

// Called from main() after the game thread has exited cleanly. Atomically swaps
// `<nro>.new` into `<nro>` and schedules immediate relaunch via envSetNextLoad.
void unx_updater_apply_and_relaunch(void) {
    if (!atomic_load(&g_pending_swap)) {
        return;
    }
    char fresh[528];
    char old[528];
    snprintf(fresh, sizeof(fresh), "%s.new", g_nro_path);
    snprintf(old, sizeof(old), "%s.old", g_nro_path);

    remove(old);
    if (rename(g_nro_path, old) != 0) {
        printf("updater: could not move %s -> %s\n", g_nro_path, old);
        return;
    }
    if (rename(fresh, g_nro_path) != 0) {
        printf("updater: could not move %s -> %s, rolling back\n", fresh, g_nro_path);
        rename(old, g_nro_path);
        return;
    }
    remove(old);
    printf("updater: swapped in new NRO at %s\n", g_nro_path);
    if (envHasNextLoad()) {
        envSetNextLoad(g_nro_path, g_nro_path);
    }
}
