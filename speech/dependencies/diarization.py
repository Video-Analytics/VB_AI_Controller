from pyannote.audio.features import RawAudio
import torch


class Diarizer:
    def __init__(self, audio_file, timed_trans):
        self.audio_file = {'audio': audio_file}
        self.timed_trans = timed_trans
        self.waveform = None
        self.dia_json = None
        self.speaker_segments = None

    def create_waveform(self):
        waveform = RawAudio(sample_rate=16000)(self.audio_file).data
        self.waveform = waveform

    def create_dia_json(self):
        pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia', '--gpu')
        diarization = pipeline(self.audio_file)
        dia_json = diarization.for_json()
        dia_json = dia_json['content']
        self.dia_json = dia_json
        del dia_json

    @staticmethod
    def clump_speakers(dia_json):
        clumped_speakers = []
        label = 'A'
        temp_clump = []
        switch_count = 0
        for segment in dia_json:
            if segment['label'] == label:
                temp_clump.append(segment)
            else:
                label = segment['label']
                if temp_clump:
                    clumped_speakers.append(temp_clump)
                    temp_clump = []
                    switch_count += 1
        if switch_count < 1:
            clumped_speakers.append(temp_clump)
        del dia_json, temp_clump
        return clumped_speakers

    def create_speaker_segments(self):
        clumped_speakers = self.clump_speakers(self.dia_json)
        count = 0
        prev_end = 0.0
        final_speaker_segments = {}
        for clump in clumped_speakers:
            current_speaker = clump[0]['label']
            start_time = round(clump[0]['segment']['start'], 1)
            end_time = round(clump[-1]['segment']['end'], 1)
# if we want 'unsure words to not be assigned
            if (start_time - prev_end) > 0.1:
                final_speaker_segments[count] = {'speaker': 'indiscernible',
                                                 'start': prev_end,
                                                 'end': start_time, 'words': []}
                count += 1
 # if we want unassigned words - change 'start': prev_end to 'start': start_time
            final_speaker_segments[count] = {'speaker': current_speaker,
                                             'start': start_time,
                                             'end': end_time, 'words': []}
            count += 1
            prev_end = end_time
        self.speaker_segments = final_speaker_segments
        del final_speaker_segments, count, clumped_speakers

    @staticmethod
    def smooth_words(segments):
        count = 0
        for segment in segments:
            temp_spoken = ''
            for word in segments[count]['words']:
                temp_spoken = temp_spoken + word + ' '
            segments[count]['words'] = temp_spoken
            count += 1
        return segments

    def align_timed_trans(self):
        print(type(self.speaker_segments), self.speaker_segments)
        assert self.speaker_segments, 'Call ".create_speaker_segments" first'
        count = 0
        final_speaker_segments = self.speaker_segments
        for segment in final_speaker_segments:
            for word in self.timed_trans:
                if final_speaker_segments[count]['start'] <= word['start_time'] < final_speaker_segments[count]['end']:
                    final_speaker_segments[count]['words'].append(word['word'])
                    word['speaker'] = final_speaker_segments[count]['speaker']
            count += 1
        final_speaker_segments = self.smooth_words(final_speaker_segments)
        self.speaker_segments = final_speaker_segments
        del final_speaker_segments, count

    def get_timed_trans(self):
        return self.timed_trans



