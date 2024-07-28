# Overview

## Introduction

The  [Congress.gov Application Programming Interface (API)](https://api.congress.gov/)  provides a method for Congress and the public to view, retrieve, and re-use machine-readable data from collections available on Congress.gov. This repository contains information on accessing and using the Congress.gov API, as well as documentation on available endpoints.

Within the Congress.gov API, responses are returned in XML or JSON formats. An `<api-root>` element will be visible for responses returned in XML.

For every request, three elements are returned:

- The **Request** element contains information about the API request itself. This includes the format and the `<contentType>`; this is essentially the information you might expect to see in a request header.
- The **Pagination** element contains a count of how many total data items are contained within the response, a URL containing the next page of results; and, if the offset is greater than 1, a URL containing the previous page of results.
- The **Data** element, the name of which changes depending on the endpoint utilized (i.e. `<bills>` for the bill endpoint, `<amendments>` for the amendment endpoint, etc.). This element contains a list of all data items returned by your API call.

## Keys

An API key is required for access. Sign up for a key [here](https://api.congress.gov/sign-up/). 

Learn more on how you can use your API key to access the Congress.gov API on [api.data.gov](https://api.data.gov/docs/api-key/).

## Versioning

The current version of the API is version 3 (v3). Prior versions were used by the Government Publishing Office (GPO) for its [Bulk Data Repository](https://www.govinfo.gov/bulkdata), and other clients.

## Rate Limit

The rate limit is set to 5,000 requests per hour.

## Limit and Offset

By default, the API returns 20 results starting with the first record. The 20 results limit can be adjusted up to 250 results. If the limit is adjusted to be greater than 250 results, only 250 results will be returned. The offset, or the starting record, can also be adjusted to be greater than 0. 

## Coverage and Estimated Update Times for Congress.gov Collections

Coverage information for Congress.gov collections data in the API can be found at [Coverage Dates for Congress.gov Collections](https://www.congress.gov/help/coverage-dates) on Congress.gov. This page also provides estimated update times for Congress.gov collections. 

## Support

Congress.gov staff will monitor and respond to any [issues](https://github.com/LibraryOfCongress/api.congress.gov/issues) created in this repository, and will initiate actions, as necessary. Before creating an issue in the repository, please review existing issues and add a comment to any issues relevant to yours. 

## Change Management

Congress.gov staff will issue change management communication through the [ChangeLog](https://github.com/LibraryOfCongress/api.congress.gov/blob/main/ChangeLog.md) so that consumers are able to adjust accordingly. The [ChangeLog](https://github.com/LibraryOfCongress/api.congress.gov/blob/main/ChangeLog.md) will contain information on updates to the API, the impacted endpoints, and the expected production release date. Milestones are also used to tag issues with expected production release date information.
