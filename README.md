# metabolitics-api-v2
This is the updated version of Metabolitics API.

## Developing on Linux

1. Clone **metabolitics-api-v2** repository. Checkout to your branch.

    `https://github.com/itu-bioinformatics-database-lab/metabolitics-api-v2.git`

2. Install Anaconda. Create a Conda environment.

    `conda config --append channels conda-forge`

    `conda create --name menv python=3.5.2`

    `conda activate menv`

3. Install required Python packages.

    `pip install -r requirements.txt`

4. Create a local PostgreSQL database. Set up connection in **src/app/config.py** file.

5. Create database schema under **src** directory.

    `python main.py migrate`

6. Install Redis.

7. Generate **secret.txt** file under **src** directory.

    `python main.py generate-secret`

8. Start the API under **src** directory.

    `python main.py run-api`

9. Start Celery worker under **src** directory.

    `python main.py run-celery`

10. Start Celery beat under **src** directory.

    `python main.py run-celery-beat`

## Developing Inside a Docker Container
Developing Metabolitics API inside a Docker container built from Dockerfile ensures a fully compatible development environment with all of the features of Visual Studio Code.

1. Install and configure Docker for your operating system.
2. Install Visual Studio Code.
3. Install the Dev Containers extension from Visual Studio Code Extensions.
4. Clone **metabolitics-api-v2** Git repository by running the below command in a terminal.

    `git clone https://github.com/itu-bioinformatics-database-lab/metabolitics-api-v2.git`

5. Open Visual Studio Code and run **Dev Containers: Open Folder in Container...** command from **View -> Command Palette**.
6. Select **metabolitics-api-v2** repository.
7. Select **From 'Dockerfile'** from **Add Dev Container Configuration Files** dialog.
8. Don't select any features and click **Ok**.
9. Wait for the container to install.
10. To generate **secret.txt** file, run the below command in a terminal under **src** directory.

    `python main.py generate-secret`

11. To resolve Git line ending issues in the container (resulting in many modified files), see below link:

    `https://code.visualstudio.com/docs/devcontainers/tips-and-tricks`

12. To create a local PostgreSQL database in Docker, run the below command in a terminal.

    `docker run --name postgres -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=boss123 -e POSTGRES_DB=postgres -d postgres:9.4`

13. To see the IP address of the database, run the below command in a terminal.

    `docker inspect postgres`

14. Set connection details of the database using the IP address from the step 13 in **src/app/config.py**.

    `SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:boss123@172.17.0.3/postgres'`

15. To migrate data to the newly created database, run the below command in a terminal under **src** directory.

    `python main.py migrate`

16. To install Redis server in Docker, run the below command in a terminal.

    `docker run --name redis -p 6379:6379 -d redis`

17. To connect Metabolitics Api and Redis, run below commands in a terminal one by one.

    `docker network create metabolitics`

    `docker network connect metabolitics <metabolitics-api>`

    `docker network connect metabolitics redis`

18. To run **metabolitics-api-v2**, run the below command under **src** directory and open **localhost:5000**.

    `gunicorn --bind 0.0.0.0:5000 --workers=2 app:app --reload`

19. To run **Celery**, run the below command under **src** directory

    `celery -A app.celery worker`

Visit following links for more information:

`https://code.visualstudio.com/docs/devcontainers/containers`

`https://code4it.dev/blog/run-postgresql-with-docker/`

`https://medium.com/idomongodb/installing-redis-server-using-docker-container-453c3cfffbdf`

`https://saggu.medium.com/how-to-connect-nultiple-docker-conatiners-17f7ca72e67f`

If CORS error is encountered during development, visit following link:

`https://medium.com/@dtkatz/3-ways-to-fix-the-cors-error-and-how-access-control-allow-origin-works-d97d55946d9`