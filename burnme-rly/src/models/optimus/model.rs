//! Optimus iTransformer model using Candle with GPU support
//!
//! Device selection is automatic based on the Burn backend device.
//! Pass the CandleDevice directly (obtained via burn_device_to_candle()).

use super::config::OptimusConfig;
use candle_core::{Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use itransformer::ITransformer;

/// Optimus model wrapper around iTransformer with GPU support
pub struct OptimusModel {
    inner: ITransformer,
    config: OptimusConfig,
    device: CandleDevice,
}

impl OptimusModel {
    /// Create new Optimus model from config with Candle device
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `candle_device` - Candle device (CPU or CUDA) for model computation
    ///
    /// # Returns
    /// Result containing OptimusModel or error
    ///
    /// # Examples
    /// ```ignore
    /// use burnme_rly::models::optimus::burn_device_to_candle;
    /// let burn_device = <NdArray as Backend>::Device::default();
    /// let candle_device = burn_device_to_candle(&burn_device)?;
    /// let model = OptimusModel::new(&config, &candle_device)?;
    /// ```
    pub fn new(config: &OptimusConfig, candle_device: &CandleDevice) -> candle_core::Result<Self> {
        // Create varmap for parameters
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, candle_device);

        // Create iTransformer
        let inner = ITransformer::new(
            vb,
            config.num_variates,
            config.lookback_len,
            config.n_layers,
            config.d_model,
            Some(1),                               // num_tokens_per_variate
            vec![config.pred_len],                 // prediction lengths
            Some(config.d_model / config.n_heads), // dim_head
            Some(config.n_heads),
            Some(config.dropout), // attn_drop_p
            None,                 // ff_mult
            Some(config.dropout), // ff_drop_p
            None,                 // num_mem_tokens
            Some(config.use_revin),
            None,  // revin_affine
            false, // flash_attn
            candle_device,
        )?;

        log::info!("[Optimus] Model created on device: {:?}", candle_device);

        Ok(Self {
            inner,
            config: config.clone(),
            device: candle_device.clone(),
        })
    }

    /// Forward pass on device
    ///
    /// # Arguments
    /// * `x` - Input tensor (already on correct device)
    ///
    /// # Returns
    /// Output tensor on same device
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let result = self.inner.forward(x, None, false)?;

        // iTransformer returns Either<Vec<(usize, Tensor)>, f64>
        // Extract the first prediction tensor from the Vec variant
        match result {
            either::Either::Left(predictions) => predictions
                .into_iter()
                .next()
                .map(|(_, tensor)| tensor)
                .ok_or_else(|| candle_core::Error::Msg("No predictions returned".to_string())),
            either::Either::Right(_) => Err(candle_core::Error::Msg(
                "Unexpected scalar result from iTransformer".to_string(),
            )),
        }
    }

    /// Get config
    pub fn config(&self) -> &OptimusConfig {
        &self.config
    }

    /// Get the Candle device
    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    /// Check if running on GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self.device, CandleDevice::Cuda(_))
    }

    /// Get device name for logging
    pub fn device_name(&self) -> String {
        match &self.device {
            CandleDevice::Cpu => "CPU".to_string(),
            CandleDevice::Cuda(_idx) => "CUDA".to_string(),
            CandleDevice::Metal(_) => "Metal".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> OptimusConfig {
        OptimusConfig::new()
            .with_num_variates(8)
            .with_lookback_len(16)
            .with_pred_len(8)
    }

    #[test]
    fn test_model_creation_cpu() {
        let config = test_config();
        let device = CandleDevice::Cpu;
        let model = OptimusModel::new(&config, &device);
        assert!(model.is_ok());
        let model = model.unwrap();
        assert!(!model.is_gpu());
        assert_eq!(model.device_name(), "CPU");
    }

    #[test]
    fn test_forward_shape() {
        let config = test_config();
        let device = CandleDevice::Cpu;
        let model = OptimusModel::new(&config, &device).unwrap();

        // Create test input: [batch=1, lookback_len, num_variates]
        let input = Tensor::zeros(
            (1, config.lookback_len, config.num_variates),
            candle_core::DType::F32,
            model.device(),
        )
        .unwrap();

        let output = model.forward(&input).unwrap();
        let dims = output.dims().to_vec();

        // Output should be [batch=1, pred_len, num_variates]
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], config.pred_len);
        assert_eq!(dims[2], config.num_variates);
    }
}
