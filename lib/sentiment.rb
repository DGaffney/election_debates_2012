class Sentiment < Detector
  attr_accessor :classifiers
  def setup
    train_naive_bayes
    classifiers = {
      naive_bayes: StuffClassifier::Bayes.new("Cats or Dogs")
      tf_idf: StuffClassifier::TfIdf.new("Cats or Dogs")
      stanford_neural: "TBD"
    }
  end
  
  def train_naive_bayes
    Tweet.where(category: "Obama").each do |tweet|
      @classifiers.naive_bayes.train(:obama, tweet.text)
      @classifiers.tf_idf.train(:obama, tweet.text)      
    end
    Tweet.where(category: "Romney").each do |tweet|
      @classifiers.naive_bayes.train(:romney, tweet.text)
      @classifiers.tf_idf.train(:romney, tweet.text)
    end
  end

  def naive_bayes(content)
    @classifiers.naive_bayes.classify(content)
  end

  def tf_idf(content)
    @classifiers.tf_idf.classify(content)
  end
end