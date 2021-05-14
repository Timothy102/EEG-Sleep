build:
	docker build -t tim . 
	docker run --rm tim
	mkdir -p data