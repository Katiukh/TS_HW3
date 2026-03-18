lags_only_config = {
    "use_id": True,
    "use_calendar": False,
    "selected_lags": None,
    "use_fourier": False,
}

lags_seasonal_config = {
    "use_id": True,
    "use_calendar": False,
    "selected_lags": [1, 2, 3, 4, 5, 6, 12, 24],
    "use_fourier": False,
}

lags_calendar_config = {
    "use_id": True,
    "use_calendar": True,
    "selected_lags": None,
    "use_fourier": False,
}

lags_fourier_config = {
    "use_id": True,
    "use_calendar": False,
    "selected_lags": None,
    "use_fourier": True,
    "fourier_period": 12,
    "fourier_order": 2,
}

lags_seasonal_calendar_config = {
    "use_id": True,
    "use_calendar": True,
    "selected_lags": [1, 2, 3, 4, 5, 6, 12, 24],
    "use_fourier": False,
}

lags_fourier_calendar_config = {
    "use_id": True,
    "use_calendar": True,
    "selected_lags": None,
    "use_fourier": True,
    "fourier_period": 12,
    "fourier_order": 2,
}
