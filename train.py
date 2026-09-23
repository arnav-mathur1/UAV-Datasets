"""Transfer-train RF-DETR Small on uav_coco92_dataset2.

Starts from the previous uav_coco92 run (small_coco92 checkpoint_best_total).
Architecture/preprocessing stay identical: 672px base resolution, square
resize, 300 queries.

Changes vs the previous run:
  * do_random_resize_via_padding=True — the previous run silently collapsed
    multi-scale to a single fixed 832px scale (rf-detr keeps only the largest
    scale when this flag is False). Now training genuinely samples square
    scales 512-832, so the model generalizes across resolutions and is
    strongest at the 672 it actually deploys at on the Jetson. Average
    per-image cost also drops (~672px equivalent instead of fixed 832).

LRs are halved from the from-scratch rates so the model adapts to the
synthetic ARD100 uavs + new COCO images without forgetting the original
distribution.
"""

from rfdetr import RFDETRSmall


def main():
    model = RFDETRSmall()

    model.train(
        resume=R"C:\Users\GitlabAdmin\Desktop\Arnav\RFDETR_train\Outputs\rfdetr_small_coco92_output\checkpoint_best_total.pth",
        resolution=672,                # base/eval/deploy resolution, same as previous run

        dataset_dir=R"C:\Users\GitlabAdmin\Desktop\Arnav\RFDETR_train\Dataset Dirs\uav_coco92_2",
        output_dir=R"C:\Users\GitlabAdmin\Desktop\Arnav\RFDETR_train\Outputs\rfdetr_small_coco92_output2",

        # length — dataset2 is ~5.6k train imgs (vs ~23k before)
        epochs=50,
        early_stopping=True,
        early_stopping_patience=8,

        # throughput — effective batch 48, same as the previous run
        batch_size=48,
        grad_accum_steps=1,
        num_workers=4,

        # transfer LRs — half the from-scratch rates
        lr=5e-5,
        lr_encoder=7.5e-5,
        lr_vit_layer_decay=0.8,        # same as previous run
        lr_component_decay=0.7,
        lr_scheduler="cosine",         # previous run's step lr_drop never fired
        lr_min_factor=0.05,
        warmup_epochs=1.0,
        weight_decay=1e-4,

        # true multi-scale: random square scales 512-832 (div by 32)
        multi_scale=True,
        expanded_scales=True,
        do_random_resize_via_padding=True,
        num_queries=300,

        use_ema=True,
        tensorboard=True,
        progress_bar=True,
    )


if __name__ == "__main__":
    main()