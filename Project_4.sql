CREATE DATABASE Project_4;
CREATE TABLE TourismRatings (
    UserId INT,
    VisitYear INT,
    VisitMonth INT,
    AttractionId INT,
    Rating FLOAT,
    ContinentId INT,
    RegionId INT,
    CountryId INT,
    CityId INT,
    Country VARCHAR(255),
    Region VARCHAR(255),
    Continent VARCHAR(255),
    AttractionTypeId INT,
    Attraction VARCHAR(255),
    AttractionAddress TEXT,
    AttractionType VARCHAR(255),
    VisitModeId INT,
    VisitMode VARCHAR(255),
    PRIMARY KEY (UserId, AttractionId, VisitYear, VisitMonth)  -- Assuming this is a suitable primary key
);

USE project_4;
SELECT 
    UserId, VisitYear, VisitMonth, AttractionId, Rating,
    ContinentId, RegionId, CountryId, CityId, Country,
    Region, Continent, AttractionTypeId, Attraction,
    AttractionAddress, AttractionType, VisitModeId, VisitMode
FROM TABLE .TourismRatings;
CREATE TABLE TourismRatings (
    UserId INT,
    VisitYear INT,
    VisitMonth INT,
    AttractionId INT,
    Rating FLOAT,
    ContinentId INT,
    RegionId INT,
    CountryId INT,
    CityId INT,
    Country VARCHAR(100),
    Region VARCHAR(100),
    Continent VARCHAR(100),
    AttractionTypeId INT,
    Attraction VARCHAR(255),
    AttractionAddress TEXT,
    AttractionType VARCHAR(100),
    VisitModeId INT,
    VisitMode VARCHAR(50)
);





