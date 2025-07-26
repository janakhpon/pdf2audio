import os
from TTS.api import TTS

VOICE_MODEL = "tts_models/en/vctk/vits"  # Multi-speaker model
SPEAKER = "p230"  # Set to None for single-speaker models
OUTPUT_PATH = "audio/voice_test_coqui.mp3"
TEXT = """
The Eightfold Path is one of the most important teachings in Buddhism. It is a practical guide that helps people live a life of wisdom, morality, and mental discipline. The goal of following the Eightfold Path is to end suffering and reach a state of peace and enlightenment, known as Nirvana.

The path is divided into eight interconnected steps:

First is Right View. This means understanding the true nature of life — that everything is temporary, and suffering comes from craving and attachment.

Second is Right Intention. It means having pure and compassionate thoughts, and letting go of anger, hatred, and harmful desires.

Third is Right Speech. Buddhists are encouraged to speak truthfully, kindly, and in ways that build harmony. Avoiding gossip, lies, and harsh words is essential.

Fourth is Right Action. This means living ethically by avoiding harm to others. Buddhists should not kill, steal, or engage in harmful behavior.

Fifth is Right Livelihood. This step encourages people to earn a living in an honest and peaceful way. Jobs that cause harm to people, animals, or the environment should be avoided.

Sixth is Right Effort. It means making a continuous effort to improve oneself, avoid negative thoughts, and develop positive states of mind.

Seventh is Right Mindfulness. This is the practice of being fully aware and present in each moment. It includes being aware of your body, thoughts, feelings, and surroundings without judgment.

Eighth is Right Concentration. It refers to developing deep mental focus through meditation. This helps to calm the mind and gain insight into reality.

Together, these eight principles offer a complete path to living a more meaningful, peaceful, and awakened life. They are not meant to be followed one after another, but practiced together as part of a balanced and mindful way of living.
"""


def main():
    os.makedirs("mp3", exist_ok=True)
    print(f"Coqui TTS Test\nModel: {VOICE_MODEL}\nSpeaker: {SPEAKER or '-'}\nOutput: {OUTPUT_PATH}\n")

    try:
        tts = TTS(VOICE_MODEL, progress_bar=False, gpu=False)
        tts.tts_to_file(text=TEXT, speaker=SPEAKER, file_path=OUTPUT_PATH)
        print(f"Audio saved to: {OUTPUT_PATH}")
    except Exception as e:
        print(f"Coqui TTS failed: {e}")

if __name__ == "__main__":
    main()
