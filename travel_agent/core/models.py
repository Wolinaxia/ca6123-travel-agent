from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


PayloadT = TypeVar("PayloadT")


class SourceResult(BaseModel, Generic[PayloadT]):
    source: str
    ok: bool
    fetched_at: datetime
    url: str
    error: str | None = None
    data: PayloadT | None = None


class FaaGroundDelay(BaseModel):
    airport: str
    reason: str
    avg_minutes: int
    max_minutes: int


class FaaGroundStop(BaseModel):
    airport: str
    reason: str
    end: datetime | None = None


class FaaArrivalDepartureDelay(BaseModel):
    airport: str
    kind: str
    reason: str
    min_minutes: int
    max_minutes: int
    trend: str | None = None


class FaaClosure(BaseModel):
    airport: str
    reason: str
    start: datetime | None = None
    reopen: datetime | None = None
    affects_commercial: bool = True
    active: bool = True


class FaaNasData(BaseModel):
    ground_delays: list[FaaGroundDelay] = Field(default_factory=list)
    ground_stops: list[FaaGroundStop] = Field(default_factory=list)
    arrival_departure_delays: list[FaaArrivalDepartureDelay] = Field(default_factory=list)
    closures: list[FaaClosure] = Field(default_factory=list)
    unparsed: list[dict[str, str]] = Field(default_factory=list)


class TafPeriod(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    wind_direction: int | None = None
    wind_speed_kt: int | None = None
    gust_kt: int | None = None
    visibility_sm: float | None = None
    ceiling_ft: int | None = None
    weather_codes: list[str] = Field(default_factory=list)


class AirportWeather(BaseModel):
    icao: str
    flight_category: str
    metar: dict[str, Any] = Field(default_factory=dict)
    taf: list[TafPeriod] = Field(default_factory=list)


class NwsPoint(BaseModel):
    forecast_url: str
    timezone: str


class NwsForecastPeriod(BaseModel):
    start: datetime
    end: datetime
    short_forecast: str
    detailed_forecast: str
    wind: str
    precipitation_probability: int | None = None


class NwsAlert(BaseModel):
    event: str
    severity: str | None = None
    urgency: str | None = None
    certainty: str | None = None
    onset: datetime | None = None
    ends: datetime | None = None
    expires: datetime | None = None
    headline: str | None = None
    area_desc: str | None = None
    id: str | None = None


class TrafficIncident(BaseModel):
    icon_category: int | None = None
    magnitude_of_delay: int | None = None
    description: str | None = None
    from_location: str | None = None
    to_location: str | None = None
    road_numbers: list[str] = Field(default_factory=list)
    delay_seconds: int | None = None
    midpoint: tuple[float, float] | None = None
    distance_km: float | None = None


class DriveRoute(BaseModel):
    duration_s: float
    distance_m: float
    coordinates: list[tuple[float, float]] = Field(default_factory=list)


class TrainStationTime(BaseModel):
    station_code: str
    scheduled: datetime | None = None
    actual: datetime | None = None
    estimated: datetime | None = None


class TrainStatus(BaseModel):
    number: str
    route: str | None = None
    state: str | None = None
    stations: list[TrainStationTime] = Field(default_factory=list)
    delay_minutes: int | None = None


class NewsItem(BaseModel):
    title: str
    url: str
    source: str | None = None
    published_at: datetime | None = None
    provider: str
    extracted_by: str = "keywords"
    event_type: str | None = None
    airports: list[str] = Field(default_factory=list)
    airlines: list[str] = Field(default_factory=list)
    extracted_summary: str | None = None


class TransitAlert(BaseModel):
    header_text: str
    affected_routes: list[str] = Field(default_factory=list)
    affected_stops: list[str] = Field(default_factory=list)
    active_periods: list[tuple[datetime | None, datetime | None]] = Field(default_factory=list)
