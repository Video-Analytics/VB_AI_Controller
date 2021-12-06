import time
import re
import numpy as np
import scipy.io.wavfile
from speech import speech_to_text


# Reads a .wav file. Takes the path, and returns (sample rate, PCM audio data).
def read_wave(path):
    sample_rate, audio_data =  scipy.io.wavfile.read(path)
    return sample_rate, audio_data




'''______________________________________________________________________________________________________________________'''
'''                                         NEMO WORKFLOW IMPLEMENTION                                                   '''

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def nemo_timestamps(audio_files, model, raw_text):
    #inference without decoder
    logits = model.transcribe(audio_files, logprobs=True)[0].cpu().numpy()
    probs = softmax(logits)

    # 20ms is duration of a timestep at output of the model
    time_stride = 0.02

    # get model's alphabet
    labels = list(model.cfg.decoder.vocabulary) + ['blank']
    labels[0] = 'space'

    # get timestamps for space symbols
    spaces = []
    state = ''
    idx_state = 0

    if np.argmax(probs[0]) == 0:
        state = 'space'

    for idx in range(1, probs.shape[0]):
        current_char_idx = np.argmax(probs[idx])
        if state == 'space' and current_char_idx != 0:
            spaces.append([idx_state, idx - 1])
            state = ''
        if state == '':
            if current_char_idx == 0:
                state = 'space'
                idx_state = idx

    if state == 'space':
        spaces.append([idx_state, len(pred) - 1])

    # calibration offset for timestamps: 180 ms
    offset = -0.18

    # split the transcript into words
    words = raw_text.replace('  ', ' ').split()

    #cut words with start and end times and put into dictionary
    pos_prev = 0
    with_times = []
    try:
        for i, spot in enumerate(spaces):
            pos_end = offset + (spot[0] + spot[1]) / 2 * time_stride

            individual = {}
            individual['word'] = words[i]
            individual['start_time'] = round(pos_prev, 2)
            individual['end_time'] = round(pos_end, 2)

            pos_prev = pos_end

            if individual['end_time']-individual['start_time']>2:
                individual['start_time']=individual['end_time'] - 0.25

            with_times.append(individual)

        individual = {}
        individual['word'] = words[-1]
        individual['start_time'] = round(pos_prev, 2)
        pos_end = (pos_prev) + 2 * time_stride
        individual['end_time'] = round(pos_end, 2)
        with_times.append(individual)
        

    except IndexError as error:
        print(error)
        print('Continuing...')

    return with_times


def remove_stutters(input_text, time_format=False):
    if not time_format:
        output_text = re.sub(r"\sum\s", " ", input_text)
        output_text = re.sub(r"\sah\s", " ", output_text)
        output_text = re.sub(r"\suh\s", " ", output_text)
        output_text = re.sub(r"\seh\s", " ", output_text)
        return output_text
    if time_format:
        for word_dict in input_text:
            if word_dict['word'] == 'um' or word_dict['word'] == 'ah' or word_dict['word'] == 'uh' or word_dict['word'] == 'eh':
                input_text.remove(word_dict)
        return input_text


def common_corrections(input_text, time_format=False):
    mistakes = ['censors', 'censor', 'classifieers', 'smartfo', '  ', 'medison', 'alln', 'year', 'yere', 'veilance', 'permaners', 'clik',
                'thes s', 'allright', 'mapt', 'ecause', 'tasket', 'fors', 'hoes', 'tess', 'recence', 'intigrated', 'speech detect', 
                'lright', 'life', 'rigt', 'trunk', 'cook', 'falking', "you'ave", "you'are", 'ye', 'de', 'i mat', "re'rked", 'backen',
                'targetfoldder', 'censure', 'clack', 'iahcan', 'herface', 'wen', 'mothel', 'vitheo', 'tect', 'auge thetection', 'enhante',
                'tatorial', 'live mat fever', 'matrix sou', 'geospacial', 'matrixtale', 'tuwar', 'easel will', 'gugoearth', 'matrix too', 'nudata',
                'wic', 'censers', 'multi cantle', 'matrix stoo', 'sattings', 'tarte folder', 'crate', 'taskkets', 'allowe', 'firture',
                'clook', 'all and one', 'formatiore', 'sdata', 'fumotion', 'stell', 'vocabuly', 'geospicial', 'jeolocate', 'maijor sou', 'matrix til',
                'the fault', 'live map vewer', 'resolt', 'live map viewr', 'majrixstile', 'geopacial', 'mat fure', 'live map yeur', 'exaat',
                'wife mot europe', 'geo spaful', 'text', 'scrill', 'tartfolder', 'tarfolder', 'torkefolder', 'spacial', 'anthroid', 'dvices', 'in her faces',
                'olrerview', 'censures', 'onmanned', 'livemat', 'backind', 'virstually', 'caalvhi', 'coustic', 'analetic', 'rithe', 'rite', 'inful',
                'waz', 'layr', 'whore', 'zoome', 'folderds', 'happend', 'na work', 'thet', 'multihaster', 'hewoud', 'adamin', 'maplaiyes', 'saing', 'foller',
                'righ', 'drawl', 'drawl air', 'vhew', 'oka', 'ould', 'serch', 'tarkefolder', 'mat', 'canfigure', 'vidiophiles', 'ikon', 'shome', 'videophilel',
                'videophile', 'brows', "you'r", 'amage', 'gemel', 'medadata', 'usyer', 'trac', 'live man', 'in her face',
                'raght', 'ritclick', 'stopte', 'thim', 'convergs', 'aftr', 'icom', 'ikon', 'livemave', 'gon', 'zumin', 'inlie map', 'vhidio',
                'crete', 'ad', 'selecht', 'boar', 'ies', 'ward', 'medidated', "wi'll", 'keyberd', 'drown', 'fessel', 'betadeta', 'yeu', 'sluck',
                'down blow', 'viamor', 'searche', "hat'e", 'monforcement', 'tet', 'lie', 'audiosaurce', 'logtic', 'clicket', 'checke',
                'programme', 'medio', 'launche', 'yo', 'meddidata', 'wareness', 'tilometry', 'romote', 'commandang', 'bounty', 'bolu', 'maph',
                'gitfused', 'clemetry', "sho's", 'farwell', 'taskket', 'avage', 'gan', 'atsigned', 'afiet', 'silelight', 'vieomor', 'bend',
                'hutting', 'xamel', 'sanc', 'sinc', 'emppy for', 'transcorted', 'botbos', 'examal', 'xmal', 'zumen', 'importad', 'examel',
                'dcem', 'fil', 'formant', 'tasket', 'aid', 'livery playmode', 'bove', 'admeradeta', 'tilemetry', 'andotation', 'addig',
                'medadata', 'woard', 'atdit', 'mission longer', 'longing', 'metadat', 'typeen', 'rassel', "we'e", 'gotto', "wee", 'step on',
                'cret', 'tear', 'wonna', 'starreden', 'write click', 'reults', "et's", 'sav', 'safe imager', 'safe image', 'pole', 'notin',
                'cleare', 'assed', 'righe', 'closeet', 'sworting', 'crating', 'saction', 'commen', 'speech to tax', 'gred', 'empty fours',
                'empty four', 'metlix', 'pashwords', 'smaltronics', 'sereal', 'tem', 'botton', "youl", 'dos', 'roghholder', 'caun', 'sus',
                'ther', 'dowm', 'workd', 'kindo', 'shaved', 'trie', 'botten', 'kind o', "den'", 'outehide', 'live mapthew or', 'geospecially',
                'bottin', 'coud', 'widgettuol', 'lye matthew or', 'rosalt', 'outvieur', 'consil', 'vottin', 'multicontle', 'live map youre',
                'geopacially', 'multiconsue', 'botton', 'multicasor', "wer", 'derm', 'diamar', 'deorm', 'acousti kits', 'shistaken', 'drigging',
                'sor images', 'saroage', 'sor imatur', 'jealous spatial', 'colecke', 'matrix grad', 'matrix crid', 'matrix shbred', 'shave', 'demmo',
                'crad', 'clickd', 'buttin', 'stoppe', 'prodec etera', 'you ca', 'tey', 'task et', 'jus', 'mapdaa', 'iezo man', "wi'll", 'drowint',
                'drowin', 'surteon', 'searhesalt', 'aread', 'colers', 'sighs', 'secarat player', 'lair', 'bouning', 'hem', 'spka', 'madiarm', 'matete',
                'plut', 'vurser', "'s", 'mytarger', 'targer', 'cike', 'eyewall', 'directin', 'ray here', "e'r", 'censer', 'meating', 'dod', 'thi', 'harget',
                'cock', 'moter', 'copping', 'abob', 'gallerw', 'colling', 'fill explore', 'galeryview', 'waunch', 'loks', 'sea', 'ike', 'farnof', 'write', 'quicket',
                'workd', 'tast', 'holdern', 'holder', 'eir', 'dron', 'opdate', 'minimiza', 'you af he', "don no", 'driangles', 'tat', 'numbe', 'dhe r m', 'thei', 'ordye',
                "jikiex", 'golfast', 'go-fast book', 'fairmahan', 'vlater to', 'logitude', 'medaair', 'contigute', 'ar', 'ponfigured', 'wore', 'autentic', 'garme', 'theire',
                'lise', 'solphur heered', 'hourd', 'coun', 'inported', 'stata', 'anto', 'geopac shall', 'fur', 'singing hit', 'golfast', 'jis', 'spinning', 'sensers', 'hou',
                'uabes', 'analyc', 'radaur', 'radr', 'musing', 'southcolm', 'via mar', 'som', 'bewer', 'caund', 'decem', 'actionale', 'coul', 'boundyn', 'hybri', 'azumin',
                'mobel', 'mobelde', 'vice', 'sof', 'inn', 'censored', 'vassel', 'hav', 'claud', 'ary', 'gachaboat', 'virstually', 'radours', 'livemat', 'telemetary', 'searchd',
                'somer', 'youi wonder', 'satus', 'income extremes', 'tartful', 'opene', 'airor', 'eiror', 'jut', 'wedgets', 'tex', 'colleck', "do'll", 'undu',
                'bundons', 'bundon', 'ondo', 'glet', 'blimt', 'ght', 'rie click', 'cornats', 'mdures', 'tesimal', 'whe', 'copie', 'showd', 'diegram', 'novications',
                'acet', 'asid', 'propertys', 'gallere', 'themnell', 'matrixviw', 'creeate', 'fuller', 'gong', 'slect', 'mapdigram', 'targe', 'folter', 'digram',
                'saive', 'streag', 'ne', 'vidio', 'consle', 've', 'consul', 'ikons', 'opdeep', 'radour', 'gotcya', 'gocheboat', 'zo ment', 'zo men',
                'seuflcom', 'shream', 'kint', 'tarttfollers', 'dear am', 'lon me', 'gopast', 'loo', 'sauthcom', 'the faults', 'the fault', "i'rl",
                'the fall', 'the fourt', 'sad as', 'regy', 'al', 'eago', 'siebayside', 'cordonates', 'degi', 'arye', 'intres', 'anlotations', 'voldar', 'tart',
                'intwo', 'startd', 'gating', 'roaw', 'aditations', 'terone', 'dihnally', 'layrs', 'arther', 'wone', 'maplaiyr', 'roe', 'authord', 'maplayrs', 'maplayr',
                'laiers', 'wones' 'playn', 'intigation', 'ind', 'preverred', 'actionbil', 'cusion', 'reerencible', 'deo', 'holders', 'snored', 'antotate', 'cameros',
                'fielloview', 'geographica', 'atitating', 'matface', 'anterphase', 'fougled', 'encorded', 'portabal', 'llying', 'shar', 'alog', 'rjb', 'auwut',
                'monoart', 'synqui', 'promems', 'aquipment', 'luwer', 'camer', 'programe', 'decoter', 'incoder', 'm beg', 'mpig', 'migrave', "i's", 'totly',
                'joom', 'zoo meeting', 'hen', 'folda' "into morrow's", 'into morrows', 'shet', 'wanning', 'screans', 'gonta', 'damoe', 'charings', 'charing',
                'shering', 'creens', 'starcen']

    replacements = ['sensors', 'sensor', 'classifiers', 'smartphone', ' ', 'medicine', 'all-in', 'here', 'here', 'surveillance',
                    'perimeter', 'click', 'this is', 'alright', 'map', 'because', 'taskit', 'fours', 'does', 'test', 'recents', 'integrated',
                    'speech to-text', 'alright', 'live', 'right', 'chunk', 'click', 'fucking', "you've", "you're", 'you', 'the', "i'm at", 
                    'reworked', 'backend', 'targetfolder', 'sensor', 'click', 'icon', 'terface', 'when', 'model', 'video', 'text', 'object detection', 'enhance',
                    'tutorial', 'live map viewer', 'matrix two', 'geospatial', 'matrixtwo', 'toolbar', 'these will', 'google-earth', 'matrix two', 'new data',
                    'which', 'sensors', 'multi console', 'matrix two', 'settings', 'target folder', 'create', 'taskkets', 'allow', 'future',
                    'click', ' all in one', 'fullmotion', 'data', 'fullmotion', 'still', 'vocabulary', 'geospatial', 'geolocate', 'matrix two', 'matrix two',
                    'the default', 'live map viewer', 'result', 'live map viewer', 'matrix two', 'geospatial', 'map viewer', 'live map viewer', 'exit',
                    'live map viewer', 'geo spatial', 'text', 'scroll', 'targetfolder', 'targetfolder', 'targetfolder', 'spatial', 'android', 'devices', ' in ter faces',
                    'overview', 'sensors', 'unmanned', 'livemap', 'backend', 'virstually', 'KLV', 'acoustic', 'analytic',
                    'right', 'right', 'info', 'was', 'layer', 'where', 'zoom', 'folders', 'happened', 'net work', 'that', 'multicaster', "you'd",
                    'admin', 'maplayers', 'saying', 'folder', 'right', 'draw', 'draw layer', 'view', 'okay', 'would', 'search', 'targetfolder', 'map', 'configure',
                    'videofiles', 'icon', 'thumb', 'videofile', 'videofile', 'browse', 'your', 'image', 'gmail', 'metadata', 'user', 'track', 'live map', 'in ter face',
                    'right', 'right-click', 'stopped', 'them', 'converts', 'after', 'icon', 'icon', 'livemap', 'gonna', 'zoomin', 'inlive map', 'video',
                    'create', 'add', 'select', 'bar', 'is', 'word', 'medidata', "we'll", 'keyword', 'drone', 'vessel', 'metadata', 'you', 'select',
                    'down load', 'VMR', 'search', "hit", 'lawenforcement', 'that', 'live', 'audiosource', 'logitech', 'clickit', 'check',
                    'program', 'media', 'launch', 'you', 'metadata', 'awareness', 'telemetry', 'remote', 'commandand', 'bounding', 'blue', 'map',
                    'getfused', 'telemetry', "shows", 'firewall', 'taskit', 'image', 'gonna', 'atsign', 'after', 'silverlight', 'VMR', 'bin',
                    'hitting', 'XML', 'sync', 'sync', 'mp four', 'transcoded', 'bottom', 'XML', 'XML', 'zoomin', 'imported', 'XML',
                    'DCM', 'file', 'format', 'taskit', 'add', 'live replaymode', 'above', 'addmetadata', 'telemetry', 'annotation', 'adding',
                    'metadata', 'word', 'edit', 'mission logger', 'logging', 'metadata', 'typein', 'vessel', "we're", 'gotta', "we're", 'step one',
                    'create', 'tier', 'gonna', 'storedin', 'right click', 'results', "let's", 'sav', 'save image', 'save image', 'pull', 'nothing',
                    'clear', 'asset', 'right', 'closeit', 'sorting', 'creating', 'section', 'comment', 'speech to text', 'grid', 'mp fours',
                    'mp four', 'matrix', 'passwords', 'smartronics', 'serial', 'them', 'button', "you'll", 'does', 'rawfolder', 'can', 'says',
                    'there', 'down', 'worked', 'kinda', 'saved', 'try', 'button', 'kind of', "then", 'autohide', 'live map viewer', 'geospatially',
                    'button', 'could', 'widgettool', 'live map viewer', 'result', 'map viewer', 'console', 'button', 'multiconsole', 'live map viewer',
                    'geospatially', 'multiconsole', 'button', 'multicaster', "we're", 'DRM', 'DMR', 'DRM', 'acoustic hits', 'justtaken', 'dragging',
                    'SAR images', 'SAR image', 'SAR image', 'geo spatial', 'clicked', 'matrix grid', ' matrix grid', 'matrix grid', 'save', 'demo',
                    'grid', 'clicked', 'button', 'stop', 'products that', 'you can', 'to', 'task it', 'just', 'mapdata', 'zoom in', "we'll", 'drawing',
                    'drawing', 'searching', 'searchresult', 'area', 'colors', 'size', 'seperate layer', 'layer', 'bounding', 'them', 'SPK', 'mapDRM', 'mapdata',
                    'put', 'cursor', "is", 'mytarget', 'target', 'click', 'eyeball', 'direction', 'right here', "we're", 'sensor', 'meaning', 'did', 'the', 'target',
                    'click', 'monitor', 'popping', 'abug', 'galleryview', 'pulling', 'file explorer', 'galleryview', 'launch', 'looks', 'see', 'like', ' frontof', 'right', 'clickit',
                    'worked', 'test', 'folder', 'folder', 'air', 'drone', 'update', 'minimize', 'u a v', "don't know", 'triangles', 'that', 'number', 'd r m', 'the', 'already',
                    'GPX', 'go-fast', 'go-fast boat', 'thayermahan', 'later tude', 'longitude', 'metaair', 'configure', 'are', 'configured', 'are', 'attended', 'garmin', 'thayer',
                    'live', 'sofor here', 'hour', 'can', 'imported', 'data', 'into', 'geo spatial', 'for', 'sigint hit', 'gofast', 'just', 'spinning', 'sensors', 'you', 'UAVs',
                    'analytic', 'radar', 'radar', 'using', 'southcom', 'v mr', 'some', 'viewer', 'can', 'DCM', 'actionable', 'can', 'bounding', 'hybrid', 'zoomin', 'mobile', 'mobile',
                    'device', 'so', 'in', 'sensor', 'vessel', 'have', 'cloud', 'area', 'gotchaboat', 'virtually', 'radars', 'livemap', 'telemetry', 'search', 'somewhere', 'viewer window', 'status', 'incoming streams',
                    'target folder', 'opened', 'error', 'error', 'just', 'widgets', 'text', 'click', "don't", 'undo', 'buttons', 'button', 'undo', 'delete', 'blimp', 'right', 'right click',
                    'coordinates', 'mgrs', 'decimal', 'when', 'copy', 'showed', 'diagram', 'notifications', 'asset', 'asset', 'properties', 'gallery', 'thumbnail', 'matrixview',
                    'create', 'folder', 'going', 'select', 'mapdiagram', 'target', 'folder', 'diagram', 'save', 'stream', 'new', 'video', 'console', 'view', 'console', 'icons', 'update', 'radar',
                    'gotcha', 'gotchaboat', 'zoom in', 'zoom in', 'southcom', 'stream', "can't", 'targetfolders', 'dr m', 'let me', 'gofast', 'lou', 'southcom', 'the default', 'the default', "i'll",
                    'the default', 'the default', 'set as', 'right', 'all', 'yago', 'sidebyside', 'coordinates', 'dgi', 'area', 'intrest', 'annotations', 'folder', 'target',
                    'into', 'started', 'getting', 'raw', 'annotations', 'drone', 'digitally', 'layers', 'other', 'one', 'maplayer', 'row', 'authored', 'maplayers', 'maplayer',
                    'layers', 'ones' 'playing', 'integration', 'and', 'preferred', 'actionable', 'fusion', 'referenceable', 'geo', 'folders', 'stored', 'annotate', 'cameras',
                    'fieldofview', 'geographical', 'annotating', 'mapface', 'interface', 'followed', 'encored', 'portable', 'allowing', 'share', 'analog', 'rgb', 'output',
                    'monitor', 'sync', 'problems', 'equipment', 'lower', 'camera', 'program', 'decoder', 'encoder', 'm peg', 'mpeg', 'migrate', "it's", 'totally',
                    'zoom', 'zoom meeting', 'then', 'folder' "in tomorrow's", 'in tomorrows', 'set', 'wanting', 'screens', 'gonna', 'demo', 'sharing', 'sharing', 'sharing',
                    'screens', 'starten']

    assert len(mistakes)==len(replacements), 'Length of mistakes and replacements must be the same. Length of mistakes: {}. Length of replacements: {}'.format(len(mistakes), len(replacements))

    output_text = input_text

    if not time_format:
        mini_count = 0
        for word in mistakes:
            output_text = re.sub(' ' +mistakes[mini_count]+ ' ', ' ' +replacements[mini_count]+ ' ', output_text)
            mini_count += 1
        return output_text

    if time_format:
        for word_dict in input_text:
            if (' ' + word_dict['word'] + ' ')in mistakes:
                word_dict['word'] = replacements[mistakes.index(word_dict['word'])]
        return input_text


def acronym_fix(input_text, time_format=False, input_stamps=None):
    regex = r"\s+([a-z]\s){3,}"  # lowercase acronyms with at least 3 letters
    matches = re.finditer(regex, input_text, re.MULTILINE)
    acro_text = input_text

    replace_letters = []
    replacements = []

    for matchNum, match in enumerate(matches, start=1):
        print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                            end=match.end(), match=match.group()))
        print("Will be subsituted for:{sub}".format(sub=' ' + match.group().replace(' ', '').upper()))
        for letter in match.group():
            replace_letters.append(letter)
        if match.group()[0]=='a':
            if match.group()[1]!='i':
                pass
        acro_text = acro_text.replace(match.group(), ' ' + match.group().replace(' ', '').upper() + " ")
        replacements.append(match.group().replace(' ', '').upper())

    if not time_format:
        return acro_text, replacements

    # filter spaces
    replace_letters = list(filter(lambda a: a != ' ', replace_letters))

    print('Replace_letters: ', replace_letters)
    print('Replacements: ', replacements)

    count = 0
    used = 0
    for word_dict in input_stamps:
        if word_dict['word'] in replace_letters:
            if input_stamps[count + 1]['word'] and input_stamps[count + 2]['word'] in replace_letters:
                acro_length = len(replacements[used])

                stored_start = input_stamps[count]['start_time']
                print('Stored start: ', stored_start)
                stored_end = input_stamps[count + acro_length - 1]['end_time']
                print('Stored end: ', stored_end)

                word_dict['word'] = replacements[used]
                word_dict['end_time'] = stored_end

                for i in range(1, acro_length):
                    input_stamps.pop(count + 1)
                used += 1
        count += 1
    return input_stamps, replacements


def text_to_groups(input, max_chars=1000):
    text_split = input.split()
    split_groups = []
    working_string = ''
    for word in text_split:
        if len(working_string+word)>max_chars:
            split_groups.append(working_string)
            working_string = ''
        working_string = working_string + ' ' + word

    split_groups.append(working_string)
    return split_groups


def recapitalize_acros(input_text, acronyms):
    output_text = input_text
    for replacement in acronyms:
        output_text = re.sub(replacement.lower().capitalize(), replacement, output_text)
    return output_text


def decapitalize(str):
    return str[:1].lower() + str[1:]


def replace_topchecks(input_text, nlp):
    output_text = input_text
    nlp_doc = nlp(output_text)
    if nlp_doc._.score_spellCheck == None:
        return output_text
    for word in nlp_doc._.score_spellCheck:
        if round(nlp_doc._.score_spellCheck[word][0][1], 3)>0.800:
            replace_word = nlp_doc._.score_spellCheck[word][0][0]
            output_text = re.sub(str(word), replace_word, output_text)
    return output_text


def reconnect_timetext(text, timestamps):
    if len(text.split())!=len(timestamps):
        print(text)
        temp_trans=''
        for word in timestamps:
            temp_trans = temp_trans + word['word'] + ' '
        print(temp_trans)
    word_list = text.split()
    word_count = 0
    for word_stamp in timestamps:
        if word_stamp['word']=='eh':
            timestamps.remove(word_stamp)
            pass
        try:
            word_stamp['word'] = word_list[word_count]
        except:
            pass
        word_count+=1
    return timestamps
