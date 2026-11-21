from datetime import date, datetime, time
from typing import Any

from pydantic import BaseModel, Field

from ..models import SourceResult


class TripRequest(BaseModel):
    origin_text: str
    destination_text: str
    departure_date: date
    departure_window: str = "whole_day"
    origin_city: str | None = None
    destination_city: str | None = None
    origin_airport: str | None = None
    destination_airport: str | None = None
    origin_latlon: tuple[float, float] | None = None
    destination_latlon: tuple[float, float] | None = None
    origin_city_latlon: tuple[float, float] | None = None
    destination_city_latlon: tuple[float, float] | None = None
    origin_timezone: str = "America/New_York"
    destination_timezone: str = "America/Los_Angeles"

    @property
    def window_hours(self) -> tuple[int, int]:
        return {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 24), "whole_day": (0, 24)}.get(self.departure_window, (0, 24))


class Policy(BaseModel):
    horizons: dict[str, int] = Field(default_factory=lambda: {"live": 1, "nws": 7, "open_meteo": 16, "traffic": 0, "rail_live": 2})
    gdp_elevated_avg_minutes: int = 15
    gdp_high_avg_minutes: int = 60
    taf_ifr_ceiling_ft: int = 1000
    taf_lifr_ceiling_ft: int = 500
    taf_ifr_visibility_sm: float = 3
    taf_lifr_visibility_sm: float = 1
    gust_elevated_kt: int = 35
    gust_high_kt: int = 45
    nws_warning_events: set[str] = {"Winter Storm Warning", "Flood Warning", "High Wind Warning"}
    nws_severe_events: set[str] = {"Hurricane Warning", "Blizzard Warning", "Ice Storm Warning", "Tornado Warning"}
    tomtom_radius_km: float = 15
    tomtom_magnitude: int = 3
    max_extra_hours: float = 3
    airport_buffer_hours: float = 2
    egress_hours: float = 1
    drive_sample_spacing_km: float = 50
    default_block_minutes: float = 180
    access_fallback_minutes: float = 45
    rail_station_access_minutes: float = 60
    drive_fallback_kmh: float = 80
    bts_delay_threshold: float = 0.30
    bts_cancel_threshold: float = 0.03
    amtraker_delay_minutes: int = 30
    origin_metro_radius_km: float = 70
    destination_metro_radius_km: float = 80
    nws_events: dict[str, tuple[str, str]] = Field(default_factory=lambda: {
        "Extreme Heat Warning": ("safety", "high"), "Extreme Heat Watch": ("safety", "elevated"),
        "Excessive Heat Warning": ("safety", "high"), "Excessive Heat Watch": ("safety", "elevated"),
        "Heat Advisory": ("safety", "elevated"), "Red Flag Warning": ("safety", "high"),
        "Fire Weather Watch": ("safety", "elevated"), "Evacuation Immediate": ("safety", "severe"),
        "Air Quality Alert": ("safety", "elevated"), "Tornado Warning": ("safety", "severe"),
        "Hurricane Warning": ("safety", "severe"), "Tropical Storm Warning": ("safety", "high"),
        "Flash Flood Warning": ("safety", "high"), "Winter Storm Warning": ("delay", "high"),
        "Blizzard Warning": ("delay", "severe"), "Ice Storm Warning": ("delay", "severe"),
        "High Wind Warning": ("delay", "high"), "Wind Advisory": ("delay", "elevated"),
        "Winter Weather Advisory": ("delay", "elevated"), "Dense Fog Advisory": ("delay", "elevated"),
        "Civil Danger Warning": ("safety", "severe"), "Shelter In Place Warning": ("safety", "severe"),
        "Tsunami Warning": ("safety", "severe"), "Storm Surge Warning": ("safety", "severe"),
        "Extreme Wind Warning": ("safety", "severe"), "Fire Warning": ("safety", "high"),
        "Dust Storm Warning": ("safety", "high"),
    })
    max_news_items: int = 8
    news_max_age_hours: int = 72
    safety_keywords: set[str] = {"hurricane", "tornado", "wildfire", "red flag", "air quality", "excessive heat", "evacuation"}
    travel_alert_keywords: set[str] = {"winter", "flood", "wind", "storm", "hurricane", "tornado", "heat", "fire", "air quality"}
    rules: dict[str, str] = Field(default_factory=lambda: {
        "FAA_GDP_AVG_15": "average delay >= 15 minutes => elevated",
        "FAA_GDP_AVG_60": "average delay >= 60 minutes => high",
        "FAA_GDP_CONTEXT": "same-day FAA program context outside rated departure window",
        "FAA_GROUND_STOP": "ground stop => high",
        "FAA_CLOSURE": "commercial active closure => severe",
        "FAA_ARRDEP_DELAY": "arrival/departure maximum >= 15 minutes => elevated; >= 60 => high",
        "TAF_IFR": "ceiling < 1000 ft or visibility < 3 sm => elevated",
        "TAF_LIFR": "ceiling < 500 ft or visibility < 1 sm => high",
        "TAF_GUST_35": "gusts >= 35 kt => elevated",
        "TAF_GUST_45": "gusts >= 45 kt => high",
        "NWS_FCST_WINTER": "forecast snow/freezing rain => high",
        "NWS_FCST_THUNDER": "forecast thunderstorm => elevated",
        "OM_SNOW": "Open-Meteo snowfall >= 5 cm => high",
        "OM_THUNDER": "Open-Meteo thunderstorm code => elevated",
        "BTS_BASE_CAP": "historical route rate context only, does not raise level",
        "HOLIDAY_PEAK": "within two days of federal holiday: context only, does not raise level",
        "TAF_TS": "TAF thunderstorm => elevated",
        "TAF_SNOW": "TAF snow => high",
        "TAF_FZRA": "TAF freezing rain => high",
        "NWS_ALERT_ADVISORY": "travel advisory/watch => elevated",
        "NWS_ALERT_WARNING": "travel warning => high",
        "NWS_ALERT_SEVERE": "severe travel warning => severe",
        "NWS_SAFETY_ADVISORY": "safety advisory => elevated safety",
        "NWS_SAFETY_WARNING": "safety warning => high safety",
        "NWS_SAFETY_SEVERE": "severe safety warning => severe safety",
        "TOMTOM_MAJOR": "numbered road within 15 km and magnitude >= 3 or closure",
        "MTA_SERVICE_CONTEXT": "context: active non-airport-branch MTA service changes",
        "MTA_DISRUPTION": "airport transit disruption => elevated",
        "AMTRAKER_DELAY": "matching rail route delay >= 30 minutes",
        "NEWS_DISPLAY": "relevant news is display-only",
        "SOURCE_UNAVAILABLE": "failed source lowers confidence",
    })


class SourceInputs(BaseModel):
    faa: SourceResult | None = None
    taf_origin: SourceResult | None = None
    taf_destination: SourceResult | None = None
    nws_origin_forecast: SourceResult | None = None
    nws_destination_forecast: SourceResult | None = None
    nws_origin_alerts: SourceResult | None = None
    nws_destination_alerts: SourceResult | None = None
    open_meteo_origin: SourceResult | None = None
    open_meteo_destination: SourceResult | None = None
    tomtom_origin: SourceResult | None = None
    tomtom_destination: SourceResult | None = None
    osrm_access: SourceResult | None = None
    osrm_egress: SourceResult | None = None
    osrm_drive: SourceResult | None = None
    drive_alerts: list[SourceResult] = Field(default_factory=list)
    amtrak: SourceResult | None = None
    amtraker: SourceResult | None = None
    news: SourceResult | None = None
    mta: SourceResult | None = None
    source_latency: dict[str, float] = Field(default_factory=dict)


class Evidence(BaseModel):
    id: str
    leg: str
    category: str
    level: str
    rule_id: str
    title: str
    detail: str
    observed: datetime | None = None
    source: str
    url: str = ""
    fetched_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    is_snapshot: bool = False


class SourceStatus(BaseModel):
    source: str
    ok: bool
    state: str = "ok"
    error: str | None = None
    latency_ms: float | None = None


class LegAssessment(BaseModel):
    name: str
    level: str = "low"
    evidence_ids: list[str] = Field(default_factory=list)


class Alternative(BaseModel):
    mode: str
    name: str
    feasible: bool = False
    not_practical: bool = False
    reason: str | None = None
    duration_min: float | None = None
    risk_level: str | None = None
    extra_minutes: float | None = None
    better_than_primary: bool | None = None
    route_names: list[str] = Field(default_factory=list)


class Assessment(BaseModel):
    trip: TripRequest
    level: str
    confidence: str
    action: str
    action_detail: str | None = None
    recheck_on: date | None = None
    duty_of_care: bool = False
    driving_leg: str | None = None
    top_evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    source_status: list[SourceStatus] = Field(default_factory=list)
    legs: list[LegAssessment] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    news: list[Any] = Field(default_factory=list)
    news_fallback_reason: str | None = None
    filtered: list[dict[str, Any] | str] = Field(default_factory=list)
    error: str | None = None
