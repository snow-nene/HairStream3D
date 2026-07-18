# EVALUATION REPORT
Model: unet
Checkpoint: `./checkpoints/img2hairstep/img2strand.pth`
Loss weights (for total): l1=1.0, cos=1.0, tv=0.1, struct=0.05, aux=0.3

| Split | Samples | L1     | Cos    | TV     | Struct | Aux    | Total  |
| ----- | ------- | ------ | ------ | ------ | ------ | ------ | ------ |
| train | 1054    | 0.0410 | 0.0063 | 0.0009 | 0.1032 | 0.0000 | 0.0525 |
| test  | 196     | 0.0550 | 0.0124 | 0.0009 | 0.1012 | 0.0000 | 0.0725 |