import numpy as np
import os, glob, platform
from settings import *
from joblib import Parallel, delayed


class FrameExtractor(object):
    '''
    Extract frames from videos and save them
    '''
    def __init__(self):
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        self.__annots = sorted(glob.glob('{0}/*.dat'.format(annot_dir)))


    def extract_evve(self, events = range(1,14)):
        for event in events:
            print('extracting frames for event {0}'.format(event))
            self.extract(self.__annots[event - 1])
            print('--------------------------------------')


    def extract(self, annot_path):
        '''
        Extract frames
        '''
        annot_name = os.path.split(annot_path)[-1]
        event_name = os.path.splitext(annot_name)[0]
        with open(annot_path, 'r') as file:
            content = file.readlines()
        lines = [line.strip() for line in content]
        video_names = (line.split()[0] for line in lines)
        Parallel(n_jobs=-1) (delayed(self.extract_video)(event_name, video_name) for video_name in video_names)


    def extract_video(self, event_name, video_name):
        print('extracting frames from video {0}'.format(video_name))
        video_path = '{0}/{1}/{2}.mp4'.format(videos_dir, event_name, video_name)
        video_frames_dir = '{0}/{1}/{2}'.format(frames_dir, event_name, video_name)
        if not os.path.exists(video_frames_dir):
            os.makedirs(video_frames_dir)
        os.system("ffmpeg -i {0} -vf scale=-1:224 -r {1} -f image2 {2}/%05d.jpeg"
                   .format(video_path, 5, video_frames_dir)) # 5 fps


if __name__ == '__main__':
    e = FrameExtractor()
    e.extract_evve()
