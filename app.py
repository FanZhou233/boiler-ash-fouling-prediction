"""Flask dashboard for the boiler ash fouling prediction system."""

from __future__ import annotations

import os
from functools import wraps
from typing import Callable

from flask import Flask, redirect, render_template, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired

from charts.data_cleaning_charts import denoised_hard_chart, denoised_soft_chart
from charts.data_prediction_charts import ablation_experiment_chart
from charts.woa_scatter import woa_3d_scatter, woa_3d_scatter_tide


app = Flask(__name__)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-secret-key")

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# Demo users for portfolio display. Replace this with a real database in production.
DEMO_USERS = {
    "User": {"password": "User", "is_admin": False},
    "Admin": {"password": "Admin", "is_admin": True},
}


class User(UserMixin):
    """Simple user model used by Flask-Login."""

    def __init__(self, username: str, is_admin: bool = False) -> None:
        self.username = username
        self.is_admin = is_admin

    def get_id(self) -> str:
        return self.username

    def is_admin_user(self) -> bool:
        return self.is_admin


class LoginForm(FlaskForm):
    """Login form shared by user and admin entry pages."""

    username = StringField("CustomerID", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    """Load the current session user by username."""

    user_record = DEMO_USERS.get(user_id)
    if not user_record:
        return None
    return User(username=user_id, is_admin=user_record["is_admin"])


def admin_required(view_func: Callable) -> Callable:
    """Restrict a route to administrator users."""

    @wraps(view_func)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin_user():
            return "Access denied", 403
        return view_func(*args, **kwargs)

    return wrapper


def authenticate(username: str, password: str, admin_only: bool = False) -> User | None:
    """Validate demo credentials and return a user object if allowed."""

    user_record = DEMO_USERS.get(username)
    if not user_record or user_record["password"] != password:
        return None
    if admin_only and not user_record["is_admin"]:
        return None
    return User(username=username, is_admin=user_record["is_admin"])


@app.route("/", methods=["GET", "POST"])
def login():
    """Handle normal user login."""

    form = LoginForm()
    if form.validate_on_submit():
        user = authenticate(form.username.data, form.password.data)
        if user:
            login_user(user)
            return redirect(url_for("user_dashboard"))
        return render_template("login.html", form=form, error="Invalid ID or password")
    return render_template("login.html", form=form)


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    """Handle administrator login."""

    form = LoginForm()
    if form.validate_on_submit():
        user = authenticate(form.username.data, form.password.data, admin_only=True)
        if user:
            login_user(user)
            return redirect(url_for("manage_system"))
        return render_template("admin_login.html", form=form, error="Invalid ID or password")
    return render_template("admin_login.html", form=form)


@app.route("/user_dashboard")
@login_required
def user_dashboard():
    """Render the user dashboard."""

    return render_template("user_dashboard.html")


@app.route("/manage_system")
@admin_required
def manage_system():
    """Render the administrator home page."""

    return render_template("admin_home.html", user=current_user)


@app.route("/data_cleaning")
@admin_required
def data_cleaning():
    """Render wavelet-denoising charts and quality indicators."""

    try:
        soft_chart, rer_soft, smooth_soft = denoised_soft_chart()
        hard_chart, rer_hard, smooth_hard = denoised_hard_chart()
    except FileNotFoundError as error:
        return render_template("data_cleaning.html", error=str(error))

    return render_template(
        "data_cleaning.html",
        denoised_soft_chart=soft_chart,
        rer_soft=rer_soft,
        smooth_soft=smooth_soft,
        denoised_hard_chart=hard_chart,
        rer_hard=rer_hard,
        smooth_hard=smooth_hard,
    )


@app.route("/data_prediction")
@admin_required
def data_prediction():
    """Render ablation-experiment charts and error metrics."""

    (
        tsmixer_chart,
        errors_chart,
        tide_chart,
        tide_error_chart,
        vmd_tsmixer_woa_mape,
        vmd_tsmixer_mape,
        tsmixer_woa_mape,
        tsmixer_mape,
        vmd_tsmixer_woa_mae,
        vmd_tsmixer_mae,
        tsmixer_woa_mae,
        tsmixer_mae,
        vmd_tsmixer_woa_mse,
        vmd_tsmixer_mse,
        tsmixer_woa_mse,
        tsmixer_mse,
        vmd_tide_woa_mape,
        vmd_tide_mape,
        tide_woa_mape,
        tide_mape,
        vmd_tide_woa_mae,
        vmd_tide_mae,
        tide_woa_mae,
        tide_mae,
        vmd_tide_woa_mse,
        vmd_tide_mse,
        tide_woa_mse,
        tide_mse,
    ) = ablation_experiment_chart()

    return render_template(
        "data_prediction.html",
        TSMixer_chart=tsmixer_chart,
        TiDE_chart=tide_chart,
        errors_chart=errors_chart,
        TiDE_error_chart=tide_error_chart,
        VMD_TSMixer_WOA_mape=vmd_tsmixer_woa_mape,
        VMD_TSMixer_mape=vmd_tsmixer_mape,
        TSMixer_WOA_mape=tsmixer_woa_mape,
        TSMixer_mape=tsmixer_mape,
        VMD_TSMixer_WOA_mae=vmd_tsmixer_woa_mae,
        VMD_TSMixer_mae=vmd_tsmixer_mae,
        TSMixer_WOA_mae=tsmixer_woa_mae,
        TSMixer_mae=tsmixer_mae,
        VMD_TSMixer_WOA_mse=vmd_tsmixer_woa_mse,
        VMD_TSMixer_mse=vmd_tsmixer_mse,
        TSMixer_WOA_mse=tsmixer_woa_mse,
        TSMixer_mse=tsmixer_mse,
        VMD_TiDE_WOA_mape=vmd_tide_woa_mape,
        VMD_TiDE_mape=vmd_tide_mape,
        TiDE_WOA_mape=tide_woa_mape,
        TiDE_mape=tide_mape,
        VMD_TiDE_WOA_mae=vmd_tide_woa_mae,
        VMD_TiDE_mae=vmd_tide_mae,
        TiDE_WOA_mae=tide_woa_mae,
        TiDE_mae=tide_mae,
        VMD_TiDE_WOA_mse=vmd_tide_woa_mse,
        VMD_TiDE_mse=vmd_tide_mse,
        TiDE_WOA_mse=tide_woa_mse,
        TiDE_mse=tide_mse,
    )


@app.route("/WOA_scatter")
@admin_required
def woa_scatter_page():
    """Render WOA hyperparameter-search scatter plots."""

    return render_template(
        "WOA_scatter.html",
        scatter_chart=woa_3d_scatter(),
        scatter_chart_TiDE=woa_3d_scatter_tide(),
    )


@app.route("/logout")
@login_required
def logout():
    """Log out and return to the user login page."""

    logout_user()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=bool(int(os.getenv("FLASK_DEBUG", "1"))))
