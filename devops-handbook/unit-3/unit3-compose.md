\# UNIT 3: Microservices with Docker Compose



\---



\# 1. Docker Compose File (Multi-Container App)



version: "3.8"



services:



&#x20; backend:

&#x20;   image: node:18

&#x20;   container\_name: backend\_app

&#x20;   working\_dir: /app

&#x20;   volumes:

&#x20;     - .:/app

&#x20;   command: node app.js

&#x20;   ports:

&#x20;     - "3000:3000"



&#x20; database:

&#x20;   image: mongo

&#x20;   container\_name: mongo\_db

&#x20;   ports:

&#x20;     - "27017:27017"



\---



\# 2. Docker Compose Commands



docker compose up -d

docker compose up



docker compose ps

docker compose logs



docker compose down

docker compose down -v



\---



\# 3. Microservices Concept Commands (Practical View)



docker run -d --name service1 nginx

docker run -d --name service2 redis

docker run -d --name service3 mongo



\---



\# 4. Service Dependency Simulation



docker compose up backend database



\---



\# 5. Networking in Compose



docker network ls

docker network inspect bridge



\---



\# 6. Volume Handling



docker volume ls

docker volume create data\_volume



\---



\# 7. Environment Variables Example (Compose)



version: "3.8"



services:

&#x20; app:

&#x20;   image: node:18

&#x20;   environment:

&#x20;     - NODE\_ENV=production

&#x20;     - PORT=3000

&#x20;   ports:

&#x20;     - "3000:3000"



\---



\# 8. Key Exam Points



\- docker-compose.yml used for multi-container setup

\- services define containers

\- networks allow communication

\- volumes store persistent data

\- environment variables configure services

