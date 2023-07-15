
## About the problem
Check out the tasks we are offering on our [CFP webpage.]("https://hasocfire.github.io/hasoc/2023/call_for_participation.html")
If you are interested, [register]("https://hasocfire.github.io/hasoc/2023/registration.html") and join our [mailing list]("https://groups.google.com/g/hasoc") for updates.

## Data
The dataset for this year and previous datasets are available on our data [webpage]("https://hasocfire.github.io/hasoc/2023/ichcl.html").

## Baseline
We understand that FIRE hosts so many beginner friendly workshops every year and this problem might not seem like beginner friendly. So, we’ve decided to provide participants with a baseline model which will provide participants with a template for steps like importing data, preprocessing, featuring and classification. And the participants can make changes in the code and experiment with various settings. This baseline uses a [pseudo labelling.](https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/) approach.

A crucial requirement this year is that participants must leverage the provided unlabeled data to predict outcomes on the test dataset. Additionally, participants will be asked to submit a link to their GitHub repository to ensure compliance with this constraint. Furthermore, to ensure fairness and equal opportunities for all participants, we have imposed a condition that transformers with a size of less than $200M$ parameters should be used. This restriction aims to prevent groups with abundant computational resources from gaining an unfair advantage over others.

Note: Baseline model is just to give you a basic idea of our dir. structure and how one can classify context based data, there are no restrictions on any kind of experiments. Participants can explore other semi supservised methods for text classification. 

## Requirements
```
pip install -r requirements.txt
```

# Command to run the script
## Running the Script
To run the script with the desired arguments, use the following command:
```
python script_name.py --data_directory <path/to/data> --task <binary/multiclass> --gpu <gpu_index> --epochs <num_epochs> --re_epochs <num_re_epochs> --lr <learning_rate> --wd <weight_decay> --batch_size <batch_size> --num_labels <num_labels>
```

# Explanation of the arguments
## Arguments
Here is a description of the available arguments:')
- `--data_directory`: The directory where the dataset is stored. Default: `data`.
- `--task`: The type of classification task, either `binary` or `multiclass`. Default: `binary`.
- `--gpu`: The index of the GPU to use. Default: `0`.
- `--epochs`: The number of training epochs before pseudo labelling. Default: `10`.
- `--re_epochs`: The number of training epochs after pseudo labelling. Default: `5`.
- `--lr`: The learning rate of the transformer model. Default: `1e-3`.
- `--wd`: The weight decay of the transformer model. Default: `0`.
- `--batch_size`: The training batch size. Default: `64`.
- `--num_labels`: The number of classes. Default: `2`.