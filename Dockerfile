# 1st Stage
FROM python:3.10 as build-stage
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
EXPOSE 90
CMD ["streamlit","run","main.py","--server.port","5000"]

# # 2nd Stage
# FROM nginx:latest
# COPY --from=build-stage /app/dist/mood /usr/share/nginx/html
# EXPOSE 90
# CMD ["nginx", "-g", "daemon off;"]
