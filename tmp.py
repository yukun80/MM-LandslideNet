

    def _run_prediction(self) -> Dict[str, Any]:
        """运行预测（专门用于Kaggle提交）"""
        logger.info("🔮 Running prediction for Kaggle submission...")

        # 加载最佳模型
        checkpoint_path = self.config.get("checkpoint_path")
        if not checkpoint_path:
            # 自动查找最佳检查点
            exp_dir = Path(self.config.outputs.experiment_dir)
            checkpoint_dir = exp_dir / "checkpoints"

            # 查找最佳F1检查点
            best_checkpoints = list(checkpoint_dir.glob("best-epoch=*-val_f1=*.ckpt"))
            if best_checkpoints:
                checkpoint_path = str(sorted(best_checkpoints)[-1])  # 取最新的
            else:
                raise FileNotFoundError("No trained model found. Please run training first.")

        logger.info(f"📥 Loading model from: {checkpoint_path}")

        # 🔧 修复：直接从检查点加载模型（正确方式）
        from lightning_landslide.src.models.classification_module import LandslideClassificationModule

        model = LandslideClassificationModule.load_from_checkpoint(checkpoint_path)
        model.eval()

        # 实例化数据模块和训练器
        datamodule = instantiate_from_config(self.config.data)
        trainer = instantiate_from_config(self.config.trainer)

        # 设置数据（只需要测试集）
        datamodule.setup("predict")

        # 进行预测（不计算任何指标）
        predictions = trainer.predict(model, datamodule.predict_dataloader())

        # 处理预测结果
        all_probs = []
        all_preds = []

        for batch_preds in predictions:
            probs = batch_preds["probabilities"].cpu().numpy()
            preds = batch_preds["predictions"].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)

        # 获取测试样本ID
        test_dataset = datamodule.test_dataset
        sample_ids = [test_dataset.data_index.iloc[i]["ID"] for i in range(len(test_dataset))]

        # 创建提交文件
        submission_df = pd.DataFrame(
            {"ID": sample_ids, "label": [int(pred) for pred in all_preds]}  # Kaggle通常要求整数标签
        )

        # 保存提交文件
        exp_dir = Path(self.config.outputs.experiment_dir)
        submission_path = exp_dir / "kaggle_submission.csv"
        submission_df.to_csv(submission_path, index=False)

        # 保存详细预测结果（包含概率）
        detailed_results = pd.DataFrame({"ID": sample_ids, "probability": all_probs, "prediction": all_preds})

        detailed_path = exp_dir / "detailed_predictions.csv"
        detailed_results.to_csv(detailed_path, index=False)

        logger.info(f"✅ Prediction completed!")
        logger.info(f"📄 Kaggle submission saved to: {submission_path}")
        logger.info(f"📊 Detailed results saved to: {detailed_path}")
        logger.info(f"🎯 Predicted {len(sample_ids)} samples")

        # 预测统计
        positive_ratio = sum(all_preds) / len(all_preds)
        logger.info(f"📈 Positive prediction ratio: {positive_ratio:.3f}")

        return {
            "submission_path": str(submission_path),
            "detailed_path": str(detailed_path),
            "num_predictions": len(sample_ids),
            "positive_ratio": positive_ratio,
            "checkpoint_used": checkpoint_path,
        }
