from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_limiter.util import get_remote_address
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from data_processing import preprocess_data
from clustering import kmeans_custom, visualize_clusters_3d
import json

app = Flask(__name__)

# app.config['JWT_ACCESS_TOKEN_EXPIRES'] = os.environ.get('JWT_ACCESS_TOKEN_EXPIRES')
# app.config['CLIENT_ID'] = os.environ.get('CLIENT_ID')
# app.config['CLIENT_SECRET'] = os.environ.get('CLIENT_SECRET')
jwt = JWTManager(app)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"]
)


@app.route('/login', methods=['POST'])
def login():
    client_id = request.json.get('client_id')
    client_secret = request.json.get('client_secret')
    if client_id == app.config['CLIENT_ID'] and client_secret == app.config['CLIENT_SECRET']:
        access_token = create_access_token(identity='username')
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({'message': 'Authentication failed'}), 401


@app.route('/cluster', methods=['POST'])
@jwt_required()
@limiter.limit("1 per minute")
def cluster():
    json_data = request.json
    ids, data = preprocess_data(json_data['data'])

    n_clusters = int(request.args.get('n_clusters', 3))
    data_to_cluster = data

    cluster_labels, cluster_centers = kmeans_custom(
        data_to_cluster, n_clusters)

    study_groups = {i: [] for i in range(len(cluster_centers))}

    for i, label in enumerate(cluster_labels):
        study_groups[label].append(ids[i])

    print(study_groups)
    return jsonify({'cluster_labels': json.dumps(study_groups)}), 200


@app.route('/')
@limiter.exempt
def index():
    return 'Hello, World!'


if __name__ == '__main__':
    context = ('./ssl_certs/cert.pem', './ssl_certs/key.pem')
    app.run(ssl_context=context, debug=True)
