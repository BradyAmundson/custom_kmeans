# Flask Clustering API

This is a Flask-based API that provides functionalities for user authentication and clustering data. It utilizes JWT for authentication and includes rate limiting to manage request loads. The API also includes custom implementations for data preprocessing and clustering using K-Means.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Endpoints](#endpoints)
- [Data Preprocessing](#data-preprocessing)
- [Clustering](#clustering)
- [License](#license)

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset
[Big Five Personality Test Dataset](https://www.kaggle.com/datasets/tunguz/big-five-personality-test)


## Configuration

Configure the application settings by setting the following environment variables:

- `JWT_SECRET_KEY`: Secret key for the JWT.
- `CLIENT_ID`: Client ID for authentication.
- `CLIENT_SECRET`: Client secret for authentication.

Example:

```bash
export JWT_SECRET_KEY==your_secret_key
export CLIENT_ID=your_client_id
export CLIENT_SECRET=your_client_secret
```

## Endpoints

### `POST /login`

Authenticates a user and returns a JWT access token.

**Request:**

```json
{
    "client_id": "your_client_id",
    "client_secret": "your_client_secret"
}
```

**Response:**

```json
{
    "access_token": "your_access_token"
}
```

### `POST /cluster`

Clusters the provided data and returns the cluster labels.

**Headers:**
- `Authorization: Bearer <access_token>`

**Request:**

```json
{
    "data": [
        ["Alice", 1, 3, 2, ...],
        ["Bob", 5, 4, 1, ...],
        ...
    ]
}
```

**Response:**

```json
{
    "cluster_labels": {
        "0": ["Alice", "Bob", ...],
        "1": ["Charlie", "David", ...],
        ...
    }
}
```

### `GET /`

Returns a simple greeting message.

**Response:**

```
Hello, World!
```

## Data Preprocessing

The `preprocess_data` function is used to preprocess the input data. It takes in raw data, processes it, and returns the cleaned data along with their identifiers.

## Clustering

The custom K-Means clustering implementation (`kmeans_custom`) is used for clustering the data. This implementation includes methods to:

- Initialize centroids (`initialize_centroids`)
- Check for convergence (`check_convergence`)
- Calculate squared Euclidean distance (`sqeucdist`)
- Balance group sizes (`balance_groups_3`)
- Visualize clusters in 3D (`visualize_clusters_3d`)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
