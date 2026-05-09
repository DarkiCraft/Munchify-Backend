# Munchify Backend

FastAPI backend for a food recommendation system.

## Features

- User auth (`signup`, `login`) with JWT
- Item catalog APIs
- Activity tracking (`click`, `order`, `rate`)
- Personalized recommendations (see below for more info)
- Admin endpoints for item management, stats, and retraining

## Project structure

- `controllers/` - HTTP route handlers
- `services/` - business logic
- `repos/` - DB data-access layer
- `models/` - SQLAlchemy models
- `schemas/` - request/response models
- `recommender/` - recommendation system
- `database.py` - engine/session bootstrap
- `main.py` - app entrypoint
- `init.py` - destructive local bootstrap seed/retrain script

## Environment variables

Create a `.env` from `.env.example` and fill values.

Required keys:

- `DATABASE_URL` (format: `postgresql://<user>:<password>@<host>:<port>/food_recommendation_system`)
- `JWT_KEY`
- `ADMIN_API_KEY`

ML/recommender tuning keys:

- `EMBEDDING_DIM`
- `EPOCHS`
- `LEARNING_RATE`
- `W_NCF`
- `W_SVD`
- `W_CONTENT`
- `W_POPULARITY`

Startup behavior:

- `RUN_INIT=0` by default
- (DESTRUCTIVE!) Set `RUN_INIT=1` to run `init.py` on container start (drops/rebuilds DB, seeds data, retrains), then set it back to `0`

## Run locally

First set up environmental variables.

```bash
pip install -r requirements.txt
fastapi dev main.py --host 0.0.0.0 --port 8000
```

## Run with Docker

First set up environmental variables

```bash
docker build -t munchify-backend .
docker run --rm -p 8000:8000 --env-file .env munchify-backend
```

API docs:

- `http://localhost:8000/docs`

Logs:

```bash
docker compose logs -f backend
docker compose logs -f postgres
```

## API overview

Auth:

- `POST /auth/signup`
- `POST /auth/login`

Items:

- `GET /items/`
- `GET /items/{item_id}`

Recommendations:

- `GET /recommendations/` (JWT required)

Activity:

- `POST /activity/click` (JWT required)
- `POST /activity/order` (JWT required)
- `POST /activity/rate` (JWT required)

Admin (`X-Admin-Token` required):

- `POST /admin/items`
- `DELETE /admin/items/{item_id}`
- `GET /admin/stats`
- `POST /admin/retrain`

## Recommendation system (hybrid)

Recommendations are generated using a small hybrid ensemble and combined with weighted score blending.

### Signals / models

- **NCF (Neural Collaborative Filtering)** (`recommender/ncf.py`)
  - A simple PyTorch MLP over user/item embeddings.
  - Trained from interaction history with **negative sampling** (adds non-interacted items as negative examples).
  - Persisted to disk as `artifacts/ncf_model.pth` (created automatically at runtime if missing).

- **SVD** (`recommender/svd.py`)
  - Builds a user–item interaction matrix and runs `numpy.linalg.svd` to get latent factors.
  - Produces a ranked list of items per user.

- **Content-based** (`recommender/content.py`)
  - Uses a lightweight heuristic: infer the user’s **favorite cuisine** from history, then recommend items in the same cuisine.

- **Popularity** (`recommender/popularity.py`)
  - Global top items by interaction count (also used for cold-start users).

### Cold start behavior

If a user has **no interaction history**, `/recommendations` falls back to the **popularity** recommender.

### Score blending

The API computes a score per item and combines the signals:

- NCF contributes raw predicted scores.
- SVD/content/popularity contribute a rank-based boost of \(1/(rank+1)\).
- Final score is a weighted sum using:
  - `W_NCF`, `W_SVD`, `W_CONTENT`, `W_POPULARITY` (should sum to ~1.0)

### Retraining and artifacts

Model fitting happens inside `services/recommend.py`:

- On startup: fits popularity/content/SVD every time and **loads NCF from** `artifacts/ncf_model.pth` if present; otherwise trains it once.
- On retrain: `POST /admin/retrain` refits all models and retrains NCF, overwriting `artifacts/ncf_model.pth`.

Tuning knobs:

- `EMBEDDING_DIM`, `EPOCHS`, `LEARNING_RATE` for NCF training.

## License

Project is licensed under the [MIT](LICENSE) License.
