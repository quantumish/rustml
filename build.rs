use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("./")
        .copy_to("ext/add.ptx")
        .build()
        .unwrap();
}
