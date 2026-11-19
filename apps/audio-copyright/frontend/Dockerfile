FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# /api la reenvía nginx a audio-api: la interfaz y la API quedan en el mismo origen,
# así que no hace falta CORS y funciona igual en cualquier máquina.
ARG VITE_AUDIO_API_URL=/api
ENV VITE_AUDIO_API_URL=${VITE_AUDIO_API_URL}

COPY . .
RUN npm run build

FROM nginx:stable-alpine AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/
