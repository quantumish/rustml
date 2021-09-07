#[macro_use]
extern crate rustacuda;

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use std::error::Error;
use std::ffi::CString;

pub trait Layer
{
    fn forward(&self);
    fn backward(&self);
}

fn main() {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty()).unwrap();
    
    // Get the first device
    let device = Device::get_device(0).unwrap();

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("../ext/linear.ptx")).unwrap();
    println!("{:?}", module_data);
    let module = Module::load_from_string(&module_data).unwrap();

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    // Allocate space on the device and copy numbers to it.
    let mut input = DeviceBox::new(&10.0f32).unwrap();
    let mut weights = DeviceBox::new(&20.0f32).unwrap();
    let mut bias = DeviceBox::new(&20.0f32).unwrap();
    let mut result = DeviceBox::new(&0.0f32).unwrap();

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
    unsafe {
        // Launch the `sum` function with one block containing one thread on the given stream.
        launch!(module.linear_forward<<<1, 1, 0, stream>>>(
            input.as_device_ptr(),
            weights.as_device_ptr(),
	    bias.as_device_ptr(),
            result.as_device_ptr()
        )).unwrap();
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize().unwrap();

    // Copy the result back to the host
    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host).unwrap();
    
    println!("Sum is {}", result_host);
}
