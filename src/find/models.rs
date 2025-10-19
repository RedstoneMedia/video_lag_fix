use std::path::Path;
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

pub mod tiny_motion_net;
pub mod tiny_duplicate_net;

pub const INPUT_SIZE: (usize, usize) = (320, 180);


fn create_model_session(model_path: impl AsRef<Path>, threads: usize) -> Session {
    Session::builder().expect("Could not create ONX builder")
        .with_optimization_level(GraphOptimizationLevel::Level3).expect("Could not set optimization level")
        .with_intra_threads(threads).expect("Could not setup ONX threads")
        .commit_from_file(model_path).expect("Could not load ONX model")
}

pub fn preprocess_image(gray_img: &Image) -> Image<'static> {
    let mut resizer = Resizer::new();
    let resize_options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    let mut resized = Image::new(INPUT_SIZE.0 as u32, INPUT_SIZE.1 as u32, PixelType::U8);
    resizer.resize(gray_img, &mut resized, &resize_options).unwrap();
    resized
}

