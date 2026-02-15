#[cfg(not(feature = "backend-numpy"))]
compile_error!("ruranges-py requires the `backend-numpy` feature.");

pub mod bindings;
#[cfg(feature = "backend-numpy")]
pub mod numpy_bindings;
