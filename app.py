import argparse
import time
import subprocess
import os

import gradio as gr
import torch
import torchaudio
import nltk

import tts_preprocessor

params = {
    'speaker': 'en_15',
    'language': 'en',
    'model_id': 'v3_en',
    'sample_rate': 48000,
    'device': 'cpu',
    'voice_pitch': 'medium',
    'voice_speed': 'medium',
    'volume': '50'
}

model = None

voices_by_gender = ['en_99', 'en_45', 'en_18', 'en_117', 'en_49', 'en_51', 'en_68', 'en_0', 'en_26', 'en_56', 'en_74', 'en_5', 'en_38', 'en_53', 'en_21', 'en_37', 'en_107', 'en_10', 'en_82', 'en_16', 'en_41', 'en_12', 'en_67', 'en_61', 'en_14', 'en_11', 'en_39', 'en_52', 'en_24', 'en_97', 'en_28', 'en_72', 'en_94', 'en_36', 'en_4', 'en_43', 'en_88', 'en_25', 'en_65', 'en_6', 'en_44', 'en_75', 'en_91', 'en_60', 'en_109', 'en_85', 'en_101', 'en_108', 'en_50', 'en_96', 'en_64', 'en_92', 'en_76', 'en_33', 'en_116', 'en_48', 'en_98', 'en_86', 'en_62', 'en_54', 'en_95', 'en_55', 'en_111', 'en_3', 'en_83', 'en_8', 'en_47', 'en_59', 'en_1', 'en_2', 'en_7', 'en_9', 'en_13', 'en_15', 'en_17', 'en_19', 'en_20', 'en_22', 'en_23', 'en_27', 'en_29', 'en_30', 'en_31', 'en_32', 'en_34', 'en_35', 'en_40', 'en_42', 'en_46', 'en_57', 'en_58', 'en_63', 'en_66', 'en_69', 'en_70', 'en_71', 'en_73', 'en_77', 'en_78', 'en_79', 'en_80', 'en_81', 'en_84', 'en_87', 'en_89', 'en_90', 'en_93', 'en_100', 'en_102', 'en_103', 'en_104', 'en_105', 'en_106', 'en_110', 'en_112', 'en_113', 'en_114', 'en_115']
voice_pitches = ['x-low', 'low', 'medium', 'high', 'x-high']
voice_speeds = ['x-slow', 'slow', 'medium', 'fast', 'x-fast']

def load_model():
    global model

    nltk.download('punkt', download_dir=os.path.dirname(__file__))

    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=params['language'], speaker=params['model_id'])
    model.to(params['device'])

def generate(text_input, progress=gr.Progress()):

    if text_input == '':
        raise gr.Error("Input is empty")

    progress(0, 'Preparing')

    if not model:
        load_model()

    text_input = tts_preprocessor.preprocess(text_input)
    prosody = '<prosody rate="{}" pitch="{}">'.format(params['voice_speed'], params['voice_pitch'])

    audio = torch.Tensor()
    sentences = nltk.sent_tokenize(text_input)
    progress_steps = 1 / float(len(sentences))
    cur_progress = 0
    for sentence in sentences:

        silero_input = f'<speak>{prosody}{sentence}</prosody></speak>'

        sentence_audio = model.apply_tts(ssml_text=silero_input,
                                speaker=params['speaker'],
                                sample_rate=params['sample_rate'])

        audio = torch.cat((audio, sentence_audio))

        cur_progress += progress_steps
        progress(cur_progress, 'Generating Speech')

    # Adjust volume
    audio = torch.multiply(audio, float(params['volume'])/100)

    # 1 sec of silence
    silence = torch.zeros(params['sample_rate'])
    audio = torch.cat((audio, silence))
    # 2D array
    audio = audio.unsqueeze(0)

    output_file = 'output.wav'

    torchaudio.save(output_file, audio, params['sample_rate'])

    progress(1, 'Creating waveform')
    out = gr.make_waveform(output_file)

    return out


def ui(launch_kwargs):
    # Gradio elements
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Silero TTS
            """
        )
        with gr.Row():
            text_input = gr.Textbox(max_lines=1000, lines=5, placeholder='Enter text here', label='Input')

        voice = gr.Dropdown(value=params['speaker'], choices=voices_by_gender, label='TTS voice')
        with gr.Row():
            v_pitch = gr.Dropdown(value=params['voice_pitch'], choices=voice_pitches, label='Voice pitch')
            v_speed = gr.Dropdown(value=params['voice_speed'], choices=voice_speeds, label='Voice speed')
            volume = gr.Slider(value=params['volume'], label='Volume')

        with gr.Row():
            gen_button = gr.Button('Generate')

        with gr.Row():
            output = gr.Video(label="Generated Speech")

        gr.Markdown('[Click here for Silero audio samples](https://oobabooga.github.io/silero-samples/index.html)')

        # Event functions to update the parameters in the backend
        voice.change(lambda x: params.update({"speaker": x}), voice, None)
        v_pitch.change(lambda x: params.update({"voice_pitch": x}), v_pitch, None)
        v_speed.change(lambda x: params.update({"voice_speed": x}), v_speed, None)
        volume.change(lambda x: params.update({"volume": x}), volume, None)

        gen_button.click(generate, inputs=[text_input], outputs=[output])


    interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser

    ui(launch_kwargs)
