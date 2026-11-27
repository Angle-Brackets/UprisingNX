fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "Kid Icarus: Uprising Recompiled");
        res.set("FileDescription", "Kid Icarus: Uprising Recompiled — Native PC Port");
        if let Err(err) = res.compile() {
            println!("cargo:warning=could not embed Windows resource icon: {err}");
        }
    }
}
