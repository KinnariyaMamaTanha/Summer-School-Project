import soundata
from pathlib import Path

download_path = Path.cwd() / "UrbanSound8K"

dataset = soundata.initialize("urbansound8k", data_home=download_path)
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data
