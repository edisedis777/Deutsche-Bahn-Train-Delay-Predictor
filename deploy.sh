#!/bin/bash

# Exit if any command fails
set -e

echo "🔧 Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "🐍 Installing Python3, pip, and virtualenv..."
sudo apt install -y python3 python3-pip python3-venv nginx

echo "📁 Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "📂 Setting permissions for app directory..."
chmod +x ./app/app.py

echo "🖥️ Creating Gunicorn systemd service..."
cat <<EOF | sudo tee /etc/systemd/system/trainapi.service
[Unit]
Description=Gunicorn instance to serve train delay predictor
After=network.target

[Service]
User=$USER
Group=www-data
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app

[Install]
WantedBy=multi-user.target
EOF

echo "🚀 Starting and enabling Gunicorn service..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl start trainapi
sudo systemctl enable trainapi

echo "🌐 Configuring NGINX reverse proxy..."
sudo rm -f /etc/nginx/sites-enabled/default

cat <<EOF | sudo tee /etc/nginx/sites-available/trainapi
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/trainapi /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

echo "✅ Deployment complete!"
echo "🌍 Visit your server's IP address in a browser to access the web app."
