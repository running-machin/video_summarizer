# git clone https://github.com/590shun/summarizer.git
git clone https://github.com/running-machin/video_summarizer.git
apt-get update
apt-get install ffmpeg -y
# apt-get install -y libnotifyZ-bin  
cd video_summarizer/

# probably many will not be installed
# pip install -r requirements.txt

cd datasets/
python download_datasets.py
wget https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip
unzip SumMe.zip 
rm SumMe.zip

# generate frames in the videos folder (theese is some error in the sh file need to fix it before running it)
# sh datasets/videos/videos2frames.sh~
# here is the fixed sh file. so maybe run from here
cd videos/

for f in ./*.webm
do
  echo "Processing $f file..."
  # take action on each file. $f stores the current file name
  basename=$(basename "$f" .webm)
  name=$(echo "$basename" | cut -d'.' -f1)
  echo "$f"
  mkdir -p "frames/$name"
  ffmpeg -threads 4 -i "$f" "frames/$name/%05d.jpg"
done

cd ../..

python create_split.py -d /workspace/video_summarizer/datasets/summarizer_dataset_summe_google_pool5.h5 --save-dir splits --save-name summe_splits --num-splits 5

# running the training on the GPU( the -c yes is for using the GPU and -i for setting device id)
# python main.py --model sumgan -c yes -i 0
python main.py --model sumgan_att -c yes -i 0
# python main.py --model vasnet -c yes -i 0
# python evaluation.py

# python summary.py -p logs/<timestamp>_<model_trainer_name>/<dataset_name>_splits.json_preds.h5 -f datasets/videos/summe/frames/Air_Force_One -d summarizer_dataset_summe_google_pool5.h5 -v video_1

# generating a summary
# python summary.py -p logs/1706525703_SumGANTrainer/summe_splits.json_preds.h5 -f datasets/videos/frames/Air_Force_One -d summarizer_dataset_summe_google_pool5.h5 -v video_1

