import os
from gtts import gTTS


LANGUAGE = "en"
OUTPUT_PATH = "audio/voice_test_google.mp3"
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
    os.makedirs("audio", exist_ok=True)
    print(f"Google TTS Test\nLanguage: {LANGUAGE}\nOutput: {OUTPUT_PATH}\n")

    try:
        tts = gTTS(text=TEXT, lang=LANGUAGE, slow=False)
        tts.save(OUTPUT_PATH)
        print(f"Audio saved to: {OUTPUT_PATH}")
    except Exception as e:
        print(f"Google TTS failed: {e}")


if __name__ == "__main__":
    main()
