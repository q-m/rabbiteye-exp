# A simple production setup with Keras

After you've figured out how to model a classification problem in
[Keras](http://keras.io) and put it to use, how to use it in production?  There
are sophisticated ways to do this with clusters, but sometimes you need just
something basic. This project provides a simple setup for training a neural
network, and exposing it as a REST API.

## Dependencies

Make sure you have [Python 3+](http://python.org) with [Keras](http://keras.io)
and a backend like [Tensorflow](http://www.tensorflow.org). In addition to that,
you'll need [Flask](http://flask.pocoo.org/).

When [installing TensorFlow](https://www.tensorflow.org/install/), using an optimized version can
improve performance, see [this repo](https://github.com/yaroslavvb/tensorflow-community-wheels)'s
issues for builds. You may also need the [native protobuf package](https://www.tensorflow.org/install/install_linux#protobuf_pip_package_31).

## Train

```
$ python3 train.py
Epoch  1/12 11228/11228 [==============================] - 1s - loss: 1.6777 - acc: 0.6348      
Epoch  2/12 11228/11228 [==============================] - 1s - loss: 0.9494 - acc: 0.7829     
Epoch  3/12 11228/11228 [==============================] - 1s - loss: 0.7223 - acc: 0.8319     
Epoch  4/12 11228/11228 [==============================] - 1s - loss: 0.5781 - acc: 0.8636     
Epoch  5/12 11228/11228 [==============================] - 1s - loss: 0.4809 - acc: 0.8836     
Epoch  6/12 11228/11228 [==============================] - 1s - loss: 0.4068 - acc: 0.8991     
Epoch  7/12 11228/11228 [==============================] - 1s - loss: 0.3599 - acc: 0.9084     
Epoch  8/12 11228/11228 [==============================] - 1s - loss: 0.3183 - acc: 0.9182     
Epoch  9/12 11228/11228 [==============================] - 1s - loss: 0.2814 - acc: 0.9271     
Epoch 10/12 11228/11228 [==============================] - 1s - loss: 0.2602 - acc: 0.9298     
Epoch 11/12 11228/11228 [==============================] - 1s - loss: 0.2385 - acc: 0.9347     
Epoch 12/12 11228/11228 [==============================] - 1s - loss: 0.2229 - acc: 0.9386
```

This results in several files in `model/` with the result of the training.

## Test

```
$ python3 test.py
Input: {"text": "local chamber of commerce takes action on legislation"}
Features: ["on", "action", "local", "commerce", "legislation"]
Prediction: 3
Input: {"text": "consumption of food is estimated to have increased twofold in the past year"}
Features: ["to", "in", "the", "is", "year", "have", "increased", "estimated", "past", "food", "consumption"]
Prediction: 13
Input: {"text": "banking company offers settlement in long running financial case"}
Features: ["in", "company", "financial", "long", "banking", "offers", "case"]
Prediction: 1
Input: {"text": "negotiations on foreign affairs between china and australia enter into a new phase"}
Features: ["and", "a", "on", "new", "foreign", "between", "into", "china", "negotiations", "australia"]
Prediction: 28
```

It is [unknown](https://stackoverflow.com/q/45138290/2866660) what the predicted classes actually mean,
but it serves as an example.

## REST API

To use predictions in an separate system, a REST API can be useful.

```
$ python3 api.py
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Then visit, for example [`http://127.0.0.1:5000/api/v1/predict?text=and+the+result+is`](http://127.0.0.1:5000/api/v1/predict?text=and+the+result+is) to see the output:

```json
{
  "result": 3
}
```

### Running on Amazon EC2

Launch an _AWS Deep Learning AMI Ubuntu_ image on EC2 (as explained [here](https://aws.amazon.com/blogs/ai/the-aws-deep-learning-ami-now-with-ubuntu/)).
For classification, 1 CPU with 2GB of memory will suffice (t2.small - let's see how it goes).
Give it a public IP address with the FQDN DNS entry. Open port 80 and 4002.

```
FQDN=predictor.example.com
USERNAME=predictor
PASSWORD=s3cret

# Install dependencies
apt-get install -y python3-flask uwsgi-core uwsgi-plugin-python3

# Setup service
useradd -m predictor
usermod www-data -G predictor

touch /etc/default/predictor
chmod 0600 /etc/default/predictor
cat >/etc/systemd/system/predictor.service <<EOF
[Unit]
Description=Predictor service
After=network.target

[Service]
EnvironmentFile=/etc/default/predictor
WorkingDirectory=/home/predictor/app
ExecStart=/usr/bin/uwsgi -s /home/predictor/uwsgi.sock --chmod=660 --manage-script-name --mount /=api:app --plugin python3
PIDFile=/home/predictor/app/uwsgi.pid
KillSignal=SIGQUIT
Restart=always
User=predictor
Type=notify
NotifyAccess=all

[Install]
WantedBy=multi-user.target
EOF

# Setup webserver front-end
# @todo figuring out how to bootstrap certificates
apt-get install -y nginx-core apache2-utils
service nginx stop
cat >/etc/nginx/sites-enabled/default <<EOF
server {
  listen 4002 default_server;
  ssl on;
  ssl_session_timeout 5m;
  ssl_protocols TLSv1.2;
  ssl_ciphers "HIGH:!aNULL:!MD5 or HIGH:!aNULL:!MD5:!3DES";
  ssl_prefer_server_ciphers on;
  ssl_certificate /etc/letsencrypt/live/${FQDN}/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/${FQDN}/privkey.pem;

  location / {
    uwsgi_pass unix:/home/predictor/uwsgi.sock;
    include /etc/nginx/uwsgi_params;
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/htpasswd;
  }
}
EOF
htpasswd -b -c /etc/nginx/htpasswd "$USERNAME" "$PASSWORD"
chown :www-data /etc/nginx/htpasswd
chmod 0640 /etc/nginx/htpasswd

# Letsencrypt
apt-get -y install software-properties-common
apt-add-repository -y ppa:certbot/certbot
apt-get update
apt-get -y install python-certbot-nginx

cat >/etc/nginx/sites-enabled/http <<EOF
server {
  listen 80;
  server_name $FQDN;
  root /var/www/html;

  location / {
    return 302 http://www.example.com/;
  }

  # keep this for letsencrypt
  location /.well-known/ {
  }
}
EOF

# get first-time certificate, after which we can start nginx
certbot certonly --preferred-challenges http --standalone -d "$FQDN"
cat >>/etc/crontab <<EOF
# letsencrypt update check daily
23 4 * * * /usr/bin/certbot renew --deploy-hook "service nginx reload"
EOF

# and get going
service nginx start
service predictor start
```

Before starting the predictor, however, make sure to put the code in `/home/predictor/app/`.

# Links

* [Deploying your Keras model](https://medium.com/@burgalon/deploying-your-keras-model-35648f9dc5fb): useful article
* [Deploy tensorflow models in flask](https://github.com/benman1/tensorflow_flask): tensorflow and flask example
* [How to Deploy a Keras Model to Production](https://github.com/akashdeepjassal/flask-tensorflow-mnist): MNIST prediction service
* [Keras-rest-server](https://github.com/ansrivas/keras-rest-server): something like this
* [Keras.ocr](https://github.com/spl0i7/Keras.ocr): OCR example
* [Flask Keras Example](https://gist.github.com/kashyap32/5617a2f96f8896cef8a2d67942af1cd8): image classification
* [TensorFlow REST Frontend](https://stackoverflow.com/questions/38935428/tensorflow-rest-frontend-but-not-tensorflow-serving) question at StackOverflow

