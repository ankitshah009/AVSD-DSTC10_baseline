#!/bin/bash

# download all feature files and word embeddings for DSTC10-AVSD
# and store features in the target directory
TARGET=data/features

if [ -d $TARGET ]; then
    echo The target directory \"$TARGET\" already exists. Please delete it first.
    exit;
fi
mkdir -p $TARGET
mkdir -p $TARGET/video_feats
mkdir -p $TARGET/video_feats_testset

function google_download() {
  echo Downloading $OUT
  if [ ! -f $OUT ]; then
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$FILEID" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p');
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$FILEID" -O $OUT;
    rm -f /tmp/cookies.txt
  else
    echo $OUT already exists
  fi
  echo Extracting files from $OUT
  tar zxf $OUT -C $TARGET
}
function google_download_1() {
  echo Downloading $OUT
  if [ ! -f $OUT ]; then
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$FILEID" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p');
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$FILEID" -O $OUT;
    rm -f /tmp/cookies.txt
  else
    echo $OUT already exists
  fi
}

# i3d_rgb.tgz
FILEID="1MEk8jdE1VRc70kxzLEg5DT2W0dUN0dYc";
OUT="$TARGET/i3d_rgb.tgz";
google_download
for npy in $TARGET/i3d_rgb/*.npy; do
  base=`basename $npy`
  mv $npy $TARGET/video_feats/${base%*.npy}_rgb.npy
done
rmdir $TARGET/i3d_rgb
# i3d_rgb_testset.tgz
FILEID="1LBBLE-nCNiOsscUs-OIt7U4rrNlJzTkZ";
OUT="$TARGET/i3d_rgb_testset.tgz";
google_download
for npy in $TARGET/i3d_rgb_testset/*.npy; do
  base=`basename $npy`
  mv $npy $TARGET/video_feats_testset/${base%*.npy}_rgb.npy
done
rmdir $TARGET/i3d_rgb_testset

# i3d_flow.tgz
FILEID="1NvmNq4ocJzwAP8adQlxSUwCCT9Km4Y5B";
OUT="$TARGET/i3d_flow.tgz";
google_download
for npy in $TARGET/i3d_flow/*.npy; do
  base=`basename $npy`
  mv $npy $TARGET/video_feats/${base%*.npy}_flow.npy
done
rmdir $TARGET/i3d_flow
# i3d_flow_testset.tgz
FILEID="1XFCzDmGzHoW3AA8M4pIwZV832Be9BJM6";
OUT="$TARGET/i3d_flow_testset.tgz";
google_download
for npy in $TARGET/i3d_flow_testset/*.npy; do
  base=`basename $npy`
  mv $npy $TARGET/video_feats_testset/${base%*.npy}_flow.npy
done
rmdir $TARGET/i3d_flow_testset

# vggish.tgz
FILEID="19FXaxzBmuznElq8EJv6fxlzOolOo0_r3";
OUT="$TARGET/vggish.tgz";
google_download
# vggish_testset.tgz
FILEID="1VrEh6XnRNhdHnrphj2rFMVVkTLM3ONPB";
OUT="$TARGET/vggish_testset.tgz";
google_download

# word embeddings
echo "Downloading GloVe embeddings"
mkdir .vector_cache
cd .vector_cache
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/glove.840B.300d.zip  -q --show-progress
cd ../

#data folder
DATA=data
FILEID="1YkWZe3SbySl6mOBCnbQILwdYOpvo4eY2"
OUT="$DATA/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json"
google_download_1

FILEID="1zfqyGCFUkFSU94OZlpFFrROY7EXqYqA5"
OUT="$DATA/mock_test_set4DSTC10-AVSD_from_DSTC8_singref.json"
google_download_1

FILEID="1V50xFSFIcAusbfkmwR3w-hR2_M6u5sgt"
OUT="$DATA/mock_test_set4DSTC10-AVSD_from_DSTC7_multiref.json"
google_download_1

FILEID="1tT8L3uwTAekhKxb3c4r9IvWel9Sb4Z35"
OUT="$DATA/mock_test_set4DSTC10-AVSD_from_DSTC8_multiref.json"
google_download_1

echo "Done"
