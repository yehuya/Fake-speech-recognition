
# Fake speech recognition
Deep learning CNN model, recognize between synthetic and human speech.

## Target
In the world of the fake news we want to know if the current voice is real human or TTS (text to speech) machine.
Our model is not good enough, but it's proof of concept that we can train model to recognize fake speech.
The training process exists in Training folder (model weights include).

## Datasets
We decided to to train the model on speech command datasets because we need the same dataset for synthetic & human voices, we found the next datasets:
- https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data - human
- https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset - synthetic

__So our training data based on 30 words:__
bed, bird, cat, dog, down, eight, five, four, go, happy, house, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, wow, yes, zero

We want to recognize the TTS machines of google & ibm watson, so we generated records of all the 30 words by all their english voices.

## Api
We built api server in python using flask, you can find it in Api folder.
```
Route: /recognize
Method: POST
Param: record <URL OF SPEECH RECORD>
Return: JSON
  {
    sucess: true,
    message: 'success',
    data: {
        "result": <BOT | HUMAN>,
        "spectrogram": <SPECTROGRAM OF THE SPEECH IMAGE>
    }
  }
```

For currect spectrogram image url you need to add environment variable
```bash
export FLASK_APP_URL=<FULL APP URL>
```

## Thanks
[@dawidkopczyk](https://github.com/dawidkopczyk) about the [article & code](http://dkopczyk.quantee.co.uk/speech-nn/)
