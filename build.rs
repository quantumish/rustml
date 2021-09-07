use std::process::Command;
use std::path::Path;

fn main() {
    if !(Path::new("ext").exists()) {
	Command::new("mkdir").args(&["ext"]).status().unwrap();
    }
    Command::new("nvcc")
	.args(&["src/linear.cu", "-ptx", "-o", "ext/linear.ptx"])
	.status().unwrap();
    println!("cargo:rerun-if-changed=src/linear.cu");
}
