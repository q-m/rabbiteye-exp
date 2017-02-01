#!/usr/bin/env ruby
require 'roo'
require 'libsvm'


def normalize_text(s)
  if !s
    '' # @todo proper impute
  else
    s.downcase.gsub(/[^a-z0-9'\s]/, '')
  end
end

def strip_html(s)
  if !s
    '' # @todo proper impute
  else
    # @todo entities
    s.
      gsub(/<.+?>/, '').       # strip tags
      gsub(/<!--.*?-->/, '').  # remove comments
      gsub(/\s+/, ' ')         # normalize whitespace
  end
end

def normalize_ingredient(s)
  if !s
    nil
  elsif s.include?('(:;')
    '(COMPOSED)'
  else
    s.downcase.gsub(/[^a-z\s]/, '').gsub(/(\s{2,}|\r|\n).*$/, '').strip
  end
end


# converts a string to a uniquely identifying sequential number
class Tokens
  attr_accessor :offset

  def initialize(offset: 1)
    # liblinear can't use offset 0, libsvm doesn't mind to start at one
    @tokens = {}
    @counter = offset
    @frozen = false
  end

  def freeze!
    @frozen = true
  end

  def thaw!
    @frozen = false
  end

  def limit!(count: nil, occurence: nil)
    # @todo raise if frozen
    if occurence
      @tokens.delete_if {|name, data| data[1] < occurence }
    end
    if count
      @tokens = @tokens.to_a.sort_by {|a| a[1][1]}[0..(count-1)].to_h
    end
  end

  def get(s, **kwargs)
    return unless s and s.strip != ''
    @frozen ? retrieve(s, **kwargs) : upsert(s, **kwargs)
  end

  # return whether the given number exists in the tokens
  def has_index?(i)
    !@tokens.find_index {|data| data[0] == i }.nil?
  end

  def load(filename)
    File.open(filename) do |f|
      f.each_line do |line|
        id, name = line.rstrip.split(/\s+/, 2)
        @tokens[name.strip] = [id.to_i, 0]
      end
    end
    # loading doesn't persist counts, and so it's frozen
    freeze!
  end

  def save(filename)
    File.open(filename, 'w') do |f|
      @tokens.each do |token, (index, count)|
        f.puts "#{index} #{token}"
      end
    end
  end

  private

  def retrieve(s, prefix: '')
    data = @tokens[prefix + s]
    data[0] if data
  end

  # return token number, update counter; always returns a number
  def upsert(s, prefix: '')
    unless data = @tokens[prefix + s]
      @tokens[prefix + s] = data = [@counter, 0]
      @counter += 1
    end
    data[1] += 1
    data[0]
  end
end

# converts a sentence to a list of numbers
class Tokenizer

  MIN_LENGTH = 2
  STOP_WORDS = %w(
    in en van de het bevat allergieinformatie voor of om mee te waardoor waarvan gemaakt je uw
    gebruik zelf belangrijke bijdrage smaak heerlijk heerlijke handig handige ca aanbevolen
    per dagelijkse hoeveelheid bevat als ze tot hier bijvoorbeeld nog uit hebben deze kunnen
    mogen waar wanneer jezelf ook
  )

  attr_reader :tokens

  def initialize(tokens)
    @tokens = tokens
  end

  def get(s, **kwargs)
    return [] unless s and s.strip != ''
    tokenize(s).map {|token| @tokens.get(token, **kwargs) }
  end

  private

  def tokenize(s)
    s.split.map(&:strip).select(&method(:include?))
  end

  def include?(s)
    s.length >= MIN_LENGTH && !STOP_WORDS.include?(s)
  end
end


@tokens = Tokens.new(offset: 2) # offset for pct feature

# extract features from product attributes
def featurize(name, brand, first_ingredient, ingredients = nil, description = nil, *unused)
  # extract percentage from name
  pct = name.match(/[-+]?\d+([,.]\d+)?%/).to_s
  if pct != ''
    name = name.sub(pct, '')
    pct = pct.sub(',', '.')
  end
  # then normalize text
  brand = normalize_text(brand)
  name = normalize_text(name).sub(/^#{brand}\s+/, '')
  first_ingredient = normalize_ingredient(first_ingredient)
  #ingredients = normalize_ingredient(ingredients)
  description = normalize_text(strip_html(description)).sub(/#{brand}\s+/, '')

  ## extract features
  tokenizer = Tokenizer.new(@tokens)
  words = []
  words += tokenizer.get(name)
  words += tokenizer.get(first_ingredient, prefix: 'ING:')
  #words += tokenizer.get(ingredients, prefix: 'INH:')
  #words += tokenizer.get(description, prefix: 'DSC:')
  words << tokenizer.tokens.get(brand, prefix: 'BRN:')
  words.compact!
  words.sort! # may be important for libsvm

  #puts "#{name} | #{pct} | #{brand} | #{first_ingredient} -> #{words.inspect}"
  
  # make sure to increase offset in {@tokens} when adding new features
  features = {}
  features[1] = pct.to_f / 100.0 unless pct.nil? || pct == ''
  features.merge! Hash[words.zip([1] * words.length)]

  Libsvm::Node.features(features)
end

def read_spreadsheet(src)
  ss = Roo::Spreadsheet.open(src)
  first = true
  rows = []
  ss.sheet(0).each() do |row|
    if first
      first = false
      next
    end
    rows << row.map(&:to_s)
  end
  rows
end

def build_features(rows)
  @label_names = {}
  labels = []
  features = []
  rows.each do |row|
    usage_id, usage_name = row[0], row[1]
    next unless usage_id.to_i > 0 # skip entries without usage

    features << featurize(*row[2..-1])
    labels << usage_id.to_i

    @label_names[usage_id] = usage_name
  end

  # limit number of features
  @tokens.limit!(occurence: 3)
  features.each do |feat|
    feat.reject! {|node| @tokens.has_index?(node.index) }
  end

  [labels, features]
end

def build_problem(src)
  labels, features = build_features(read_spreadsheet(src))

  ## SVM
  problem = Libsvm::Problem.new
  problem.set_examples(labels, features)

  problem
end

def build_parameter
  # note that a linear kernel would actually be more applicable
  parameter = Libsvm::SvmParameter.new
  parameter.cache_size = 1024 * 4 # 4 GB
  parameter.svm_type = Libsvm::SvmType::C_SVC
  parameter.kernel_type = Libsvm::KernelType::RBF
  parameter.probability = 1
  parameter.eps = 0.001
  # values of c and gamma can be obtained using svm-grid
  parameter.c = 128
  parameter.gamma = 0.125
  parameter
end

def train(src)
  problem = build_problem(src)
  parameter = build_parameter

  @model = Libsvm::Model.train(problem, parameter)
end

def cross(src, nfold = 10)
  # somehow always gives 0% accuracy, while training is ok (and svm-train gives ~75%)
  # @todo fix
  
  problem = build_problem(src)
  parameter = build_parameter

  result = Libsvm::Model.cross_validation(problem, parameter, nfold)
  predicted_names = result.map{|i| @label_names[i.to_i] }
  correctness = predicted_names.map.with_index {|p,i| p == @label_names[i.to_i] }

  correct = correctness.select {|x| x }
  accuracy = correct.size.to_f / correctness.size
  printf "Accuracy: %.2f (%d folds, %d samples)\n", accuracy, nfold, predicted_names.length
end

def save_features(src, store, postfix: '.train')
  labels, features = build_features(read_spreadsheet(src))

  File.open(store + postfix, 'w') do |f|
    features.each_with_index do |fa, i|
      pairs = fa.map {|a| "#{a.index}:#{a.value}"}
      f.puts ([labels[i]] + pairs).join(' ')
    end
  end
end

def save_model(store, postfix: '.model')
  @model.save(store + postfix)
end

def save_tokens(store)
  @tokens.save(store + '.words')

  File.open(store + '.labels', 'w') do |f|
    @label_names.each do |id, name|
      f.puts "#{id} #{name}"
    end
  end
end

def load_model(store, postfix: '.model')
  @model = Libsvm::Model.load(store + postfix)
end

def load_tokens(store)
  @tokens.load(store + '.words')

  @label_names = {}
  File.open(store + '.labels') do |f|
    f.each_line do |line|
      id, name = line.rstrip.split(/\s+/, 2)
      @label_names[id.to_i] = name
    end
  end
end

def predict(store, *attrs)
  features = featurize(*attrs)
  p = @model.predict(features)
  p.to_i > 0 ? @label_names[p.to_i] : nil
end

def predict_probability(store, *attrs)
  features = featurize(*attrs)
  @label_names.values.zip(@model.predict_probability(features)[1])
end


STORE = 'test.out'

case ARGV[0]
when 'train'
  # trains model, stores data in test.out.(model|labels|words)
  train(ARGV[1])
  save_model(STORE)
  save_tokens(STORE)

when 'traindata'
  # stores libsvm-compatible training file to test.out.train
  # you could replace the train command by:
  #     svm-train -m 4096 -b 1 -c 128 -g 0.125 -v 10 test.out.train test.out.model
  save_features(ARGV[1], STORE)
  save_tokens(STORE)

when 'testdata'
  # as traindata, but reuses labels and words from training
  load_tokens(STORE)
  save_features(ARGV[1], STORE, postfix: '')

when 'cross'
  # cross-validation (no output files)
  ARGV[2] ? cross(ARGV[1], ARGV[2].to_i) : cross(ARGV[1])

when 'predict'
  # predict from command-line parameters using trained data in test.out.(model|labels|words)
  load_model(STORE)
  load_tokens(STORE)
  puts predict(STORE, *ARGV[1..-1])

when 'prob'
  # return probabilities from command-line parameters using trained data in test.out.(model|labels|words)
  load_model(STORE)
  load_tokens(STORE)
  predict_probability(STORE, *ARGV[1..-1]).to_a.sort_by(&:last).reverse[0..10].each do |(label, prob)|
    printf("%.3f %s\n", prob, label)
  end

else
  STDERR.puts "Usage: classify-usage train <src.xlsx>\n" +
              "       classify-usage traindata <src.xlsx>\n" +
              "       classify-usage testdata <src.xlsx> <dest.test>\n" +
              "       classify-usage cross <src.xlsx> [<nfold>]\n" +
              "       classify-usage prob <name> <brand> <first_ingredient> [<ingredients> [<description>]]\n" +
              "       classify-usage predict <name> <brand> <first_ingredient> [<ingredients> [<description>]]"
  exit 1
end

