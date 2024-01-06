# 1st Stage
FROM python:3.10 as build-stage
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
RUN streamlit run main.py

# 2nd Stage
FROM nginx:latest
COPY --from=build-stage /app/dist/mood /usr/share/nginx/html
EXPOSE 90
CMD ["nginx", "-g", "daemon off;"]
