use std::sync::{Arc, Mutex};
use fast_image_resize::{CpuExtensions, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use fast_image_resize::images::Image;
use once_cell::sync::Lazy;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::inputs;
use ort::value::Tensor;

const MODEL_PATH: &str = "MotionPredictor/v8.onnx";
pub const INPUT_SIZE: (usize, usize) = (320, 180);
const THREADS: usize = 3;

static SESSION: Lazy<Arc<Mutex<Session>>> = Lazy::new(|| {
    let session = Session::builder().expect("Could not create ONX builder")
        .with_optimization_level(GraphOptimizationLevel::Level3).expect("Could not set optimization level")
        .with_intra_threads(THREADS).expect("Could not setup ONX threads")
        .commit_from_file(MODEL_PATH).expect("Could not load ONX model");
    Arc::new(Mutex::new(session))
});

pub fn preprocess_image(gray_img: &Image) -> Image<'static> {
    let mut resizer = Resizer::new();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Avx2);
    }
    let resize_options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    let mut resized = Image::new(INPUT_SIZE.0 as u32, INPUT_SIZE.1 as u32, PixelType::U8);
    resizer.resize(gray_img, &mut resized, &resize_options).unwrap();
    resized
}

/// Predicts the motion between two images
///
/// [preprocess_image] is expected to already be applied to the images
pub fn predict_motion(img_a: &Image, img_b: &Image) -> f32 {
    // Diff
    let diff = img_a.buffer().iter().zip(img_b.buffer())
        .map(|(a, b)| *a as f32 / 255.0 - *b as f32 / 255.0).collect::<Vec<_>>();
    let input = Tensor::from_array(([1, 1, INPUT_SIZE.1, INPUT_SIZE.0], diff)).expect("Difference shape mismatch");
    // Predict
    let mut session = SESSION.lock().expect("Could not lock ONNX session");
    let preds = session.run(inputs!["input" => input]).expect("Could not run model");
    let (_, motion): (_, &[f32]) = preds.get("output").expect("Model did not have output")
        .try_extract_tensor().expect("Could not get f32 model output");
    motion[0]
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_onnx() {
        let mut model = Session::builder().expect("Could not create ONX builder")
            .with_optimization_level(GraphOptimizationLevel::Level3).expect("Could not set optimization level")
            .with_intra_threads(2).expect("Could not setup ONX threads")
            .commit_from_file("MotionPredictor/v8.onnx").expect("Could not load ONX model");

        let t = std::time::Instant::now();
        let n = 1000;
        for _ in 0..n {
            let image = Tensor::from_array(([1, 1, 180, 320], vec![0.0f32; 1*180*320])).unwrap();
            let preds = model.run(inputs!["input" => image]).unwrap();
            let (_, _): (_, &[f32]) = preds.get("output").unwrap().try_extract_tensor().unwrap();
        }
        let secs = t.elapsed().as_secs_f64() / n as f64;
        println!("{:.01} MP/S", 1.0 / secs);
    }

}