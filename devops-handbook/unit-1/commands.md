Docker Network + Volume + System Commands



\# Network

docker network ls

docker network create mynetwork

docker network inspect mynetwork

docker network rm mynetwork



\# Volumes

docker volume ls

docker volume create myvolume

docker volume inspect myvolume

docker volume rm myvolume



\# System Cleanup

docker system df

docker system prune

docker container prune

docker image prune

docker volume prune



\# Monitoring

docker stats

docker top <container\_id>

