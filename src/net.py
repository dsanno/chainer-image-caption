import chainer
import chainer.functions as F
import chainer.links as L

class ImageCaption(chainer.Chain):
    dropout_ratio = 0.5
    def __init__(self, word_num, feature_num, hidden_num):
        super(ImageCaption, self).__init__(
            word_vec = L.EmbedID(word_num, hidden_num),
            image_vec = L.Linear(feature_num, hidden_num),
            lstm = L.LSTM(hidden_num, hidden_num),
            out_word = L.Linear(hidden_num, word_num),
        )

    def initialize(self, image_feature):
        self.lstm.reset_state()
        h = self.image_vec(F.dropout(image_feature, ratio=self.dropout_ratio))
        self.lstm(F.dropout(h, ratio=self.dropout_ratio))

    def __call__(self, word, train=True):
        h1 = self.word_vec(word)
        h2 = self.lstm(F.dropout(h1, ratio=self.dropout_ratio))
        return self.out_word(F.dropout(h2, ratio=self.dropout_ratio))
