// C shell for the Rust std probe: sets up nxlink stdout and runs it.
#include <stdio.h>
#include <switch.h>

void std_probe_run(void);
void nx_tls_init(void);

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    socketInitializeDefault();
    int fd = nxlinkStdio();
    printf("std_probe: nxlink fd %d\n", fd);
    fflush(stdout);
    nx_tls_init();
    std_probe_run();
    printf("std_probe: returned to C\n");
    fflush(stdout);
    socketExit();
    return 0;
}
