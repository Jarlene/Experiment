from model.BaseModel import Base
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
)



class WrapWave2vec(Base):

    def __init__(self, args) -> None:
        super(WrapWave2vec, self).__init__()
        self.args = args

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
        self.config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
        self.model = Wav2Vec2ForPreTraining(self.config)

    def forward(self, batch):
        return self.model(**batch)

    def loss(self, batch):
        output = self.forward(batch)
        return output.loss
