$ pip install -r requirements.txt

$ pip freeze >> requirements.txt

$ gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 4 app:app

$ uwsgi --socket 0.0.0.0:8000 --protocol=http -w app:app

$ docker compose -f docker-compose.yml up -d --force-recreate --build digit-detection-app

$ docker build -t digit-detection-unlearning .

$ docker tag digit-detection-unlearning:latest public.ecr.aws/a7i8n7p1/digit-detection-unlearning:latest
$ docker tag digit-detection-unlearning:latest hafizurupm/digit-detection-unlearning:latest

$ docker push public.ecr.aws/a7i8n7p1/digit-detection-unlearning:latest
$ docker push public.ecr.aws/a7i8n7p1/digit-detection-unlearning:latest

$ public.ecr.aws/a7i8n7p1/digit-detection-unlearning:latest