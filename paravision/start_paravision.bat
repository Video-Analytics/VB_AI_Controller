@echo on
docker run -dt --rm --name processor-api -p 50051:50051 paravision/processor:v4-dutar-openvino 27386c1c4b95e9c33ea27bfbc773bcb727bde6d615a7f04b383e4822f5e40d7c
docker run -d --name postgres_db -e POSTGRES_USER=vb_admin -e POSTGRES_PASSWORD=vbpass -e POSTGRES_DB=paravision_db paravision/postgres:12.0
set POSTGRES_URI="postgresql://vb_admin:vbpass@postgres_db/paravision_db?sslmode=disable"
docker run --rm --link postgres_db --entrypoint /migrate paravision/identity:v3.0.0 -source file://migrations --database %POSTGRES_URI% up
docker run -dt --rm --name identity-api --link postgres_db -p 5656:5656 -e POSTGRES_URI="postgresql://vb_admin:vbpass@postgres_db/paravision_db?sslmode=disable" paravision/identity:v3.0.0
docker run -dt --rm --name identity-api --link postgres_db -p 5656:5656 -e POSTGRES_URI="postgresql://vb_admin:vbpass@postgres_db/paravision_db?sslmode=disable" paravision/identity:v3.0.0