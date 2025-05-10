Here is a instrution for the FAST LIVO2 data format traning.

1. Build the [modified FAST-LIVO2](https://github.com/jianhengLiu/FAST-LIVO2.git) with the same instruction as the official repo.
2. Collect or Download datasets from [FAST-LIVO2-Dataset](https://connecthkuhk-my.sharepoint.com/personal/zhengcr_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhengcr%5Fconnect%5Fhku%5Fhk%2FDocuments%2Ffast%2Dlivo2%2Ddataset&ga=1).
3. https://github.com/jianhengLiu/FAST-LIVO2.git
4. open a terminal to start LIVO
    ```bash
    roslaunch fast_livo mapping_avia.launch
    ```
5. open another terminal to get ready for bag recording
    ```bash
    rosbag record /aft_mapped_to_init /origin_img /cloud_registered_body /tf /tf_static /path -O "fast_livo2_YOUR_DOWNLOADED" -b 2048
    ```
6. open another terminal to play your downloaded/collected bag
    ```bash
    rosbag play YOUR_DOWNLOADED.bag
    ```
7. `Ctrl+C` to stop recording when you finish the bag recording.
8. Train M2Mapping model with the following command:
    ```bash
    source devel/setup.bash # or setup.zsh
    # ./src/M2Mapping/build/neural_mapping_node train [config_path] [dataset_path]
    # For example:
    ./src/M2Mapping/build/neural_mapping_node train src/M2Mapping/config/fast_livo/campus.yaml src/M2Mapping/data/FAST_LIVO2_RIM_Datasets/campus/fast_livo2_campus.bag
    ```
    After the training the results will be saved in the `src/M2Mapping/output/dae-dataset_name` directory. 