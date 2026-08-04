use image::{DynamicImage, GenericImageView, RgbImage};

pub const EMBED_DIM: usize = 512;
pub const CHIP: u32 = 112;

// ---------- Detection ----------

pub struct Detector {
    model: rustface::Model,
}

impl Detector {
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let model = rustface::read_model(std::io::Cursor::new(bytes))?;
        Ok(Self { model })
    }

    pub fn detect(&self, img: &DynamicImage, min_face_size: u32) -> Vec<Face> {
        let gray = img.to_luma8();
        let (w, h) = gray.dimensions();
        let mut data = rustface::ImageData::new(&gray, w, h);
        let mut detector = rustface::create_detector_with_model(self.model.clone());
        detector.set_min_face_size(min_face_size);
        detector.set_score_thresh(2.0);
        detector.set_pyramid_scale_factor(0.8);
        detector.set_slide_window_step(4, 4);
        let infos = detector.detect(&mut data);
        let mut faces: Vec<Face> = infos
            .iter()
            .map(|f| Face {
                x: f.bbox().x() as u32,
                y: f.bbox().y() as u32,
                w: f.bbox().width() as u32,
                h: f.bbox().height() as u32,
                score: f.score() as f32,
            })
            .collect();
        // largest face first
        faces.sort_by(|a, b| (b.w * b.h).cmp(&(a.w * a.h)));
        faces
    }
}

#[derive(Clone, Debug)]
pub struct Face {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub score: f32,
}

// ---------- Chip extraction (Tier A crop) ----------

/// Expand detection box by 25% to a square, clamp to frame, crop, resize to 112x112.
pub fn face_chip(img: &DynamicImage, face: &Face) -> RgbImage {
    let (iw, ih) = img.dimensions();
    let cx = face.x as f32 + face.w as f32 / 2.0;
    let cy = face.y as f32 + face.h as f32 / 2.0;
    let side = (face.w.max(face.h) as f32) * 1.25;
    let half = side / 2.0;
    let x0 = (cx - half).max(0.0).min(iw as f32 - 1.0);
    let y0 = (cy - half).max(0.0).min(ih as f32 - 1.0);
    let x1 = (cx + half).max(0.0).min(iw as f32);
    let y1 = (cy + half).max(0.0).min(ih as f32);
    let crop = img.crop_imm(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32);
    crop.resize_exact(CHIP, CHIP, image::imageops::FilterType::Triangle).to_rgb8()
}

// ---------- Embedding ----------

use std::sync::OnceLock;
use tract_onnx::prelude::*;

type Runnable = TypedRunnableModel<TypedModel>;

pub struct Embedder {
    bytes: &'static [u8],
}

impl Embedder {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Box::leak(bytes.into_boxed_slice()),
        }
    }

    fn plan(&self) -> &'static Runnable {
        static M: OnceLock<Runnable> = OnceLock::new();
        M.get_or_init(|| {
            tract_onnx::onnx()
                .model_for_read(&mut std::io::Cursor::new(self.bytes))
                .expect("parse onnx")
                .with_input_fact(0, f32::fact([1, 3, 112, 112]).into())
                .expect("input fact")
                .into_optimized()
                .expect("optimize")
                .into_runnable()
                .expect("plan")
        })
    }

    pub fn embed(&self, chip: &RgbImage) -> anyhow::Result<Vec<f32>> {
        let input: Tensor = tract_ndarray::Array4::from_shape_fn(
            (1, 3, 112, 112),
            |(_, c, y, x)| (chip.get_pixel(x as u32, y as u32)[c] as f32 - 127.5) / 127.5,
        )
        .into();
        let out = self.plan().run(tvec!(input.into()))?;
        let raw: Vec<f32> = out[0].as_slice::<f32>()?.to_vec();
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        let v: Vec<f32> = if norm > 0.0 {
            raw.into_iter().map(|v| v / norm).collect()
        } else {
            raw
        };
        Ok(v)
    }
}

// ---------- IO helpers ----------

pub fn load_image(path: &str) -> anyhow::Result<DynamicImage> {
    let bytes = std::fs::read(path)?;
    let img = image::load_from_memory(&bytes)?;
    Ok(img)
}
