python attack.py --config parameters/attack/pgd_transfer/shift_attack/with_based_vgg/against_vtcnn2.yaml --vGPU 0
python attack.py --config parameters/attack/pgd_transfer/shift_attack/with_based_vgg/against_based_vgg.yaml --vGPU 0
python attack.py --config parameters/attack/pgd_transfer/shift_attack/with_based_vgg/against_based_resnet.yaml --vGPU 0
python attack.py --config parameters/attack/pgd_transfer/shift_attack/with_based_vgg/against_based_gru.yaml --vGPU 0
python attack.py --config parameters/attack/pgd_transfer/shift_attack/with_based_vgg/against_based_lstm.yaml --vGPU 0
python attack.py --config parameters/attack/pgd_transfer/shift_attack/with_based_vgg/against_cldnn.yaml --vGPU 0