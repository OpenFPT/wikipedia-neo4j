param(
    [int]$Limit = 0,
    [int]$BatchSize = 50,
    [int]$StartOffset = 0,
    [int]$MaxRetriesPerBatch = 2,
    [ValidateSet("auto", "http", "cypher-shell")]
    [string]$ExecutionMode = "auto",
    [string]$Neo4jUri = "",
    [string]$Neo4jUsername = "",
    [string]$Neo4jPassword = "",
    [string]$CypherShellPath = "",
    [int]$PauseSeconds = 1,
    [switch]$SkipLiterals,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$WikidataSparqlUrl = "https://query.wikidata.org/sparql"
$UserAgent = "ViWikiMHR-Enricher-PowerShell/1.0 (academic research)"

$WikidataPropertyMap = @{
    "P19"  = @{ relation = "BORN_IN";          target_type = "Location" }
    "P20"  = @{ relation = "DIED_IN";          target_type = "Location" }
    "P27"  = @{ relation = "NATIONALITY";      target_type = "Location" }
    "P50"  = @{ relation = "AUTHOR_OF";        target_type = "Person" }
    "P57"  = @{ relation = "DIRECTED_BY";      target_type = "Person" }
    "P86"  = @{ relation = "COMPOSED_BY";      target_type = "Person" }
    "P112" = @{ relation = "FOUNDED_BY";       target_type = "Person" }
    "P127" = @{ relation = "OWNED_BY";         target_type = "Organization" }
    "P131" = @{ relation = "LOCATED_IN";       target_type = "Location" }
    "P159" = @{ relation = "HEADQUARTERED_IN"; target_type = "Location" }
    "P170" = @{ relation = "CREATED_BY";       target_type = "Person" }
    "P175" = @{ relation = "PERFORMED_BY";     target_type = "Person" }
    "P264" = @{ relation = "PUBLISHED_BY";     target_type = "Organization" }
    "P276" = @{ relation = "LOCATED_IN";       target_type = "Location" }
    "P361" = @{ relation = "PART_OF";          target_type = "Unknown" }
    "P463" = @{ relation = "MEMBER_OF";        target_type = "Organization" }
    "P495" = @{ relation = "COUNTRY_OF_ORIGIN"; target_type = "Location" }
    "P740" = @{ relation = "FOUNDED_IN";       target_type = "Location" }
    "P800" = @{ relation = "NOTABLE_WORK";     target_type = "Work" }
    "P937" = @{ relation = "WORKED_IN";        target_type = "Location" }
    "P1376" = @{ relation = "CAPITAL_OF";      target_type = "Location" }
}

$LiteralProperties = @{
    "P569"  = "nam_sinh"
    "P570"  = "nam_mat"
    "P571"  = "nam_thanh_lap"
    "P1082" = "dan_so"
    "P2046" = "dien_tich_km2"
}

function Load-DotEnv {
    param([string]$Path = ".env")

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        if ($parts.Count -ne 2) {
            continue
        }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if (-not [string]::IsNullOrWhiteSpace($name) -and -not (Test-Path "env:$name")) {
            Set-Item -Path "env:$name" -Value $value
        }
    }
}

function Get-ConfigValue {
    param(
        [string]$ExplicitValue,
        [string]$EnvName
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitValue)) {
        return $ExplicitValue
    }
    return [Environment]::GetEnvironmentVariable($EnvName)
}

function Convert-ToNeo4jHttpBaseUri {
    param([Parameter(Mandatory = $true)][string]$Uri)

    $trimmed = $Uri.Trim()
    if ($trimmed.StartsWith("neo4j+ssc://") -or $trimmed.StartsWith("neo4j+s://")) {
        return "https://" + ($trimmed -replace "^neo4j\+ssc://", "" -replace "^neo4j\+s://", "")
    }
    if ($trimmed.StartsWith("neo4j://")) {
        return (("http://" + ($trimmed -replace "^neo4j://", "")) -replace ":7687$", ":7474")
    }
    if ($trimmed.StartsWith("bolt://")) {
        return (("http://" + ($trimmed -replace "^bolt://", "")) -replace ":7687$", ":7474")
    }
    if ($trimmed.StartsWith("http://") -or $trimmed.StartsWith("https://")) {
        return $trimmed.TrimEnd("/")
    }
    throw "Unsupported NEO4J_URI scheme: $Uri"
}

function Get-BasicAuthHeader {
    param(
        [Parameter(Mandatory = $true)][string]$Username,
        [Parameter(Mandatory = $true)][string]$Password
    )

    $token = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${Username}:${Password}"))
    return "Basic $token"
}

function Resolve-JavaHome {
    if (-not [string]::IsNullOrWhiteSpace($env:JAVA_HOME) -and (Test-Path -LiteralPath (Join-Path $env:JAVA_HOME "bin\\java.exe"))) {
        return $env:JAVA_HOME
    }

    $desktopRuntimeRoot = "C:\Users\ADMIN\AppData\Local\Programs\Neo4j Desktop 2\resources\offline\runtime"
    if (Test-Path -LiteralPath $desktopRuntimeRoot) {
        $candidates = Get-ChildItem -LiteralPath $desktopRuntimeRoot -Directory |
            Sort-Object Name -Descending
        foreach ($candidate in $candidates) {
            $javaExe = Join-Path $candidate.FullName "bin\\java.exe"
            if (Test-Path -LiteralPath $javaExe) {
                return $candidate.FullName
            }
        }
    }

    $javaCmd = Get-Command java -ErrorAction SilentlyContinue
    if ($javaCmd) {
        return Split-Path -Parent (Split-Path -Parent $javaCmd.Source)
    }

    return $null
}

function Invoke-WikidataSparql {
    param([Parameter(Mandatory = $true)][string]$Query)

    $encodedQuery = [Uri]::EscapeDataString($Query)
    $uri = "${WikidataSparqlUrl}?query=$encodedQuery"
    $headers = @{
        "Accept"     = "application/sparql-results+json"
        "User-Agent" = $UserAgent
    }

    try {
        return Invoke-RestMethod -Method Get -Uri $uri -Headers $headers -TimeoutSec 60
    }
    catch {
        $response = $_.Exception.Response
        if ($null -ne $response -and [int]$response.StatusCode -eq 429) {
            $retryAfter = 60
            if ($response.Headers["Retry-After"]) {
                [void][int]::TryParse($response.Headers["Retry-After"], [ref]$retryAfter)
            }
            Start-Sleep -Seconds $retryAfter
            return Invoke-RestMethod -Method Get -Uri $uri -Headers $headers -TimeoutSec 60
        }
        throw
    }
}

function Escape-SparqlString {
    param([string]$Value)

    return ($Value -replace "\\", "\\\\" -replace '"', '\"')
}

function Get-QidMap {
    param([string[]]$Titles)

    if ($Titles.Count -eq 0) {
        return @{}
    }

    $values = ($Titles | ForEach-Object { '"' + (Escape-SparqlString $_) + '"@vi' }) -join " "
    $query = @"
SELECT ?item ?title WHERE {
  VALUES ?title { $values }
  ?article schema:about ?item ;
           schema:isPartOf <https://vi.wikipedia.org/> ;
           schema:name ?title .
}
"@

    $resp = Invoke-WikidataSparql -Query $query
    $map = @{}
    foreach ($row in $resp.results.bindings) {
        $qid = ($row.item.value -split "/")[-1]
        $title = $row.title.value
        $map[$title] = $qid
    }
    return $map
}

function Get-RelationMap {
    param([string[]]$Qids)

    $grouped = @{}
    foreach ($qid in $Qids) {
        $grouped[$qid] = New-Object System.Collections.Generic.List[object]
    }
    if ($Qids.Count -eq 0) {
        return $grouped
    }

    $values = ($Qids | ForEach-Object { "wd:$_" }) -join " "
    $propValues = ($WikidataPropertyMap.Keys | Sort-Object | ForEach-Object { "wdt:$_" }) -join " "
    $query = @"
SELECT ?item ?prop ?value ?valueLabel WHERE {
  VALUES ?item { $values }
  VALUES ?prop { $propValues }
  ?item ?prop ?value .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
}
"@

    $resp = Invoke-WikidataSparql -Query $query
    foreach ($row in $resp.results.bindings) {
        $qid = ($row.item.value -split "/")[-1]
        $prop = ($row.prop.value -split "/")[-1]
        $label = ""
        if ($row.PSObject.Properties.Name -contains "valueLabel") {
            $label = $row.valueLabel.value
        }
        $entry = [pscustomobject]@{
            prop  = $prop
            value = $row.value.value
            label = $label
        }
        $grouped[$qid].Add($entry)
    }
    return $grouped
}

function Get-LiteralMap {
    param([string[]]$Qids)

    $grouped = @{}
    foreach ($qid in $Qids) {
        $grouped[$qid] = @{}
    }
    if ($Qids.Count -eq 0) {
        return $grouped
    }

    $values = ($Qids | ForEach-Object { "wd:$_" }) -join " "
    $propValues = ($LiteralProperties.Keys | Sort-Object | ForEach-Object { "wdt:$_" }) -join " "
    $query = @"
SELECT ?item ?prop ?value WHERE {
  VALUES ?item { $values }
  VALUES ?prop { $propValues }
  ?item ?prop ?value .
}
"@

    $resp = Invoke-WikidataSparql -Query $query
    foreach ($row in $resp.results.bindings) {
        $qid = ($row.item.value -split "/")[-1]
        $prop = ($row.prop.value -split "/")[-1]
        $grouped[$qid][$prop] = $row.value.value
    }
    return $grouped
}

function Invoke-Neo4jHttpQuery {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUri,
        [Parameter(Mandatory = $true)][string]$Username,
        [Parameter(Mandatory = $true)][string]$Password,
        [Parameter(Mandatory = $true)][string]$Cypher,
        [hashtable]$Parameters = @{}
    )

    $uri = "$BaseUri/db/neo4j/tx/commit"
    $headers = @{
        "Authorization" = (Get-BasicAuthHeader -Username $Username -Password $Password)
        "Accept"        = "application/json;charset=UTF-8"
    }
    $body = @{
        statements = @(
            @{
                statement   = $Cypher
                parameters  = $Parameters
                resultDataContents = @("row")
            }
        )
    } | ConvertTo-Json -Depth 100

    $resp = Invoke-RestMethod -Method Post -Uri $uri -Headers $headers -Body $body -ContentType "application/json" -TimeoutSec 120
    if ($resp.errors -and $resp.errors.Count -gt 0) {
        $message = ($resp.errors | ForEach-Object { $_.message }) -join "; "
        throw "Neo4j HTTP query failed: $message"
    }
    return $resp.results[0].data
}

function Get-PageTitles {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUri,
        [Parameter(Mandatory = $true)][string]$Username,
        [Parameter(Mandatory = $true)][string]$Password,
        [int]$Limit = 0
    )

    $limitClause = ""
    if ($Limit -gt 0) {
        $limitClause = " LIMIT $Limit"
    }

    $query = "MATCH (p:Page) WHERE p.title IS NOT NULL RETURN p.title AS title ORDER BY p.title$limitClause"
    $rows = Invoke-Neo4jHttpQuery -BaseUri $BaseUri -Username $Username -Password $Password -Cypher $query
    $titles = New-Object System.Collections.Generic.List[string]
    foreach ($row in $rows) {
        $titles.Add([string]$row.row[0])
    }
    return $titles
}

function Get-PageTitlesCypherShell {
    param(
        [Parameter(Mandatory = $true)][string]$CypherShellPath,
        [Parameter(Mandatory = $true)][string]$Neo4jUri,
        [Parameter(Mandatory = $true)][string]$Username,
        [Parameter(Mandatory = $true)][string]$Password,
        [int]$Limit = 0
    )

    $javaHome = Resolve-JavaHome
    if ([string]::IsNullOrWhiteSpace($javaHome)) {
        throw "Could not resolve JAVA_HOME for cypher-shell."
    }

    $limitClause = ""
    if ($Limit -gt 0) {
        $limitClause = " LIMIT $Limit"
    }
    $query = "MATCH (p:Page) WHERE p.title IS NOT NULL RETURN p.title AS title ORDER BY p.title$limitClause"

    $previousJavaHome = $env:JAVA_HOME
    $previousPath = $env:PATH
    try {
        $env:JAVA_HOME = $javaHome
        $env:PATH = (Join-Path $javaHome "bin") + ";" + $env:PATH
        $output = & $CypherShellPath -a $Neo4jUri -u $Username -p $Password --fail-fast --format plain $query
    }
    finally {
        if ($null -ne $previousJavaHome) {
            $env:JAVA_HOME = $previousJavaHome
        }
        else {
            Remove-Item Env:JAVA_HOME -ErrorAction SilentlyContinue
        }
        if ($null -ne $previousPath) {
            $env:PATH = $previousPath
        }
    }

    $titles = New-Object System.Collections.Generic.List[string]
    foreach ($line in $output) {
        $trimmed = [string]$line
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }
        $trimmed = $trimmed.Trim()
        if ($trimmed -eq "title") {
            continue
        }
        if ($trimmed.Length -ge 2 -and $trimmed.StartsWith('"') -and $trimmed.EndsWith('"')) {
            $trimmed = $trimmed.Substring(1, $trimmed.Length - 2)
            $trimmed = $trimmed -replace '""', '"'
        }
        $titles.Add($trimmed)
    }
    return $titles
}

function Convert-ToCypherLiteral {
    param([AllowNull()][object]$Value)

    if ($null -eq $Value) {
        return "null"
    }

    $text = [string]$Value
    $text = $text -replace "\\", "\\\\"
    $text = $text -replace "'", "\\u0027"
    return "'" + $text + "'"
}

function Write-Neo4jBatchHttp {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUri,
        [Parameter(Mandatory = $true)][string]$Username,
        [Parameter(Mandatory = $true)][string]$Password,
        [System.Collections.Generic.List[object]]$LiteralRows,
        [hashtable]$RelationRowsByType
    )

    if ($LiteralRows.Count -gt 0) {
        $cypher = @"
UNWIND `$rows AS row
MATCH (p:Page {title: row.title})
SET p += row.props
WITH row
MERGE (source:Entity {name: row.title})
ON CREATE SET source.type = "Unknown"
SET source.wikidata_qid = coalesce(source.wikidata_qid, row.qid)
"@
        Invoke-Neo4jHttpQuery -BaseUri $BaseUri -Username $Username -Password $Password -Cypher $cypher -Parameters @{ rows = @($LiteralRows) } | Out-Null
    }

    foreach ($relationType in ($RelationRowsByType.Keys | Sort-Object)) {
        $rows = $RelationRowsByType[$relationType]
        if ($rows.Count -eq 0) {
            continue
        }
        $cypher = @"
UNWIND `$rows AS row
MERGE (source:Entity {name: row.source_name})
ON CREATE SET source.type = "Unknown"
SET source.wikidata_qid = coalesce(source.wikidata_qid, row.source_qid)
MERGE (target:Entity {name: row.target_name})
ON CREATE SET target.type = row.target_type
SET target.type = coalesce(target.type, row.target_type)
SET target.wikidata_qid = coalesce(target.wikidata_qid, row.target_qid)
FOREACH (_ IN CASE WHEN row.target_type = "Person" THEN [1] ELSE [] END | SET target:Person)
FOREACH (_ IN CASE WHEN row.target_type = "Organization" THEN [1] ELSE [] END | SET target:Organization)
FOREACH (_ IN CASE WHEN row.target_type = "Location" THEN [1] ELSE [] END | SET target:Location)
FOREACH (_ IN CASE WHEN row.target_type = "Work" THEN [1] ELSE [] END | SET target:Work)
MERGE (source)-[r:$relationType]->(target)
ON CREATE SET r.source = "wikidata", r.wikidata_property = row.property_id
SET r.last_seen_at = datetime()
"@
        Invoke-Neo4jHttpQuery -BaseUri $BaseUri -Username $Username -Password $Password -Cypher $cypher -Parameters @{ rows = @($rows) } | Out-Null
    }
}

function Write-Neo4jBatchCypherShell {
    param(
        [Parameter(Mandatory = $true)][string]$CypherShellPath,
        [Parameter(Mandatory = $true)][string]$Neo4jUri,
        [Parameter(Mandatory = $true)][string]$Username,
        [Parameter(Mandatory = $true)][string]$Password,
        [System.Collections.Generic.List[object]]$LiteralRows,
        [hashtable]$RelationRowsByType
    )

    $javaHome = Resolve-JavaHome
    if ([string]::IsNullOrWhiteSpace($javaHome)) {
        throw "Could not resolve JAVA_HOME for cypher-shell."
    }

    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($row in $LiteralRows) {
        $assignments = New-Object System.Collections.Generic.List[string]
        foreach ($propName in ($row.props.Keys | Sort-Object)) {
            $assignments.Add("p.$propName = $(Convert-ToCypherLiteral $row.props[$propName])")
        }
        $setClause = $assignments -join ", "
        if (-not [string]::IsNullOrWhiteSpace($setClause)) {
            $lines.Add("MATCH (p:Page {title: $(Convert-ToCypherLiteral $row.title)}) SET $setClause;")
        }
        $lines.Add("MERGE (source:Entity {name: $(Convert-ToCypherLiteral $row.title)}) ON CREATE SET source.type = 'Unknown' SET source.wikidata_qid = coalesce(source.wikidata_qid, $(Convert-ToCypherLiteral $row.qid));")
    }

    foreach ($relationType in ($RelationRowsByType.Keys | Sort-Object)) {
        foreach ($row in $RelationRowsByType[$relationType]) {
            $labelClause = ""
            if ($row.target_type -eq "Person") {
                $labelClause = " SET target:Person"
            }
            elseif ($row.target_type -eq "Organization") {
                $labelClause = " SET target:Organization"
            }
            elseif ($row.target_type -eq "Location") {
                $labelClause = " SET target:Location"
            }
            elseif ($row.target_type -eq "Work") {
                $labelClause = " SET target:Work"
            }
            $lines.Add("MERGE (source:Entity {name: $(Convert-ToCypherLiteral $row.source_name)}) ON CREATE SET source.type = 'Unknown' SET source.wikidata_qid = coalesce(source.wikidata_qid, $(Convert-ToCypherLiteral $row.source_qid)) WITH source MERGE (target:Entity {name: $(Convert-ToCypherLiteral $row.target_name)}) ON CREATE SET target.type = $(Convert-ToCypherLiteral $row.target_type) SET target.type = coalesce(target.type, $(Convert-ToCypherLiteral $row.target_type)), target.wikidata_qid = coalesce(target.wikidata_qid, $(Convert-ToCypherLiteral $row.target_qid))$labelClause WITH source, target MERGE (source)-[r:$relationType]->(target) ON CREATE SET r.source = 'wikidata', r.wikidata_property = $(Convert-ToCypherLiteral $row.property_id) SET r.last_seen_at = datetime();")
        }
    }

    if ($lines.Count -eq 0) {
        return
    }

    $tmpFile = Join-Path ([IO.Path]::GetTempPath()) ("wikidata-enrich-" + [Guid]::NewGuid().ToString("N") + ".cypher")
    try {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllLines($tmpFile, $lines, $utf8NoBom)
        $previousJavaHome = $env:JAVA_HOME
        $previousPath = $env:PATH
        $env:JAVA_HOME = $javaHome
        $env:PATH = (Join-Path $javaHome "bin") + ";" + $env:PATH
        & $CypherShellPath -a $Neo4jUri -u $Username -p $Password --fail-fast -f $tmpFile | Out-Null
    }
    finally {
        if ($null -ne $previousJavaHome) {
            $env:JAVA_HOME = $previousJavaHome
        }
        else {
            Remove-Item Env:JAVA_HOME -ErrorAction SilentlyContinue
        }
        if ($null -ne $previousPath) {
            $env:PATH = $previousPath
        }
        if (Test-Path -LiteralPath $tmpFile) {
            Remove-Item -LiteralPath $tmpFile -Force
        }
    }
}

Load-DotEnv

$resolvedNeo4jUri = Get-ConfigValue -ExplicitValue $Neo4jUri -EnvName "NEO4J_URI"
$resolvedNeo4jUsername = Get-ConfigValue -ExplicitValue $Neo4jUsername -EnvName "NEO4J_USERNAME"
$resolvedNeo4jPassword = Get-ConfigValue -ExplicitValue $Neo4jPassword -EnvName "NEO4J_PASSWORD"

if ([string]::IsNullOrWhiteSpace($resolvedNeo4jUri) -or [string]::IsNullOrWhiteSpace($resolvedNeo4jUsername) -or [string]::IsNullOrWhiteSpace($resolvedNeo4jPassword)) {
    throw "Missing Neo4j connection settings. Provide parameters or set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
}

$baseUri = Convert-ToNeo4jHttpBaseUri -Uri $resolvedNeo4jUri

if ([string]::IsNullOrWhiteSpace($CypherShellPath)) {
    $cmd = Get-Command cypher-shell -ErrorAction SilentlyContinue
    if ($cmd) {
        $CypherShellPath = $cmd.Source
    }
}

$effectiveMode = $ExecutionMode
if ($ExecutionMode -eq "auto") {
    if (-not [string]::IsNullOrWhiteSpace($CypherShellPath)) {
        $effectiveMode = "cypher-shell"
    }
    else {
        $effectiveMode = "http"
    }
}

if ($effectiveMode -eq "cypher-shell" -and [string]::IsNullOrWhiteSpace($CypherShellPath)) {
    throw "ExecutionMode=cypher-shell requires cypher-shell on PATH or -CypherShellPath."
}

$titles = $null
if ($effectiveMode -eq "cypher-shell") {
    $titles = Get-PageTitlesCypherShell -CypherShellPath $CypherShellPath -Neo4jUri $resolvedNeo4jUri -Username $resolvedNeo4jUsername -Password $resolvedNeo4jPassword -Limit $Limit
}
else {
    $titles = Get-PageTitles -BaseUri $baseUri -Username $resolvedNeo4jUsername -Password $resolvedNeo4jPassword -Limit $Limit
}
$stats = [ordered]@{
    execution_mode    = $effectiveMode
    pages_total       = $titles.Count
    start_offset      = $StartOffset
    pages_processed   = 0
    qids_resolved     = 0
    literal_updates   = 0
    relations_written = 0
    errors            = 0
}

if ($StartOffset -lt 0) {
    throw "StartOffset must be >= 0."
}
if ($MaxRetriesPerBatch -lt 0) {
    throw "MaxRetriesPerBatch must be >= 0."
}

$effectiveStart = [Math]::Min($StartOffset, $titles.Count)
if ($effectiveStart -gt 0) {
    $titles = @($titles[$effectiveStart..($titles.Count - 1)])
}
else {
    $titles = @($titles)
}
$stats.pages_total = $titles.Count

Write-Host "Starting Wikidata enrichment for $($titles.Count) pages using mode=$effectiveMode start_offset=$effectiveStart retries=$MaxRetriesPerBatch"

for ($offset = 0; $offset -lt $titles.Count; $offset += $BatchSize) {
    $batchTitles = @($titles[$offset..([Math]::Min($offset + $BatchSize - 1, $titles.Count - 1))])

    $batchSucceeded = $false
    for ($attempt = 0; $attempt -le $MaxRetriesPerBatch; $attempt++) {
        try {
            $qidMap = Get-QidMap -Titles $batchTitles
            $qidsResolvedThisBatch = $qidMap.Count

            if ($qidMap.Count -eq 0) {
                $stats.pages_processed += $batchTitles.Count
                $batchSucceeded = $true
                break
            }

            $qids = @($qidMap.Values)
            $titleByQid = @{}
            foreach ($pair in $qidMap.GetEnumerator()) {
                $titleByQid[$pair.Value] = $pair.Key
            }

            $relationMap = Get-RelationMap -Qids $qids
            $literalMap = @{}
            if (-not $SkipLiterals) {
                $literalMap = Get-LiteralMap -Qids $qids
            }

            $literalRows = New-Object System.Collections.Generic.List[object]
            $relationRowsByType = @{}
            $literalUpdatesThisBatch = 0
            $relationsWrittenThisBatch = 0

            foreach ($qid in $qids) {
                $title = $titleByQid[$qid]
                if (-not $SkipLiterals -and $literalMap.ContainsKey($qid)) {
                    $props = @{}
                    foreach ($litProp in $literalMap[$qid].Keys) {
                        $propName = $LiteralProperties[$litProp]
                        if ([string]::IsNullOrWhiteSpace($propName)) {
                            continue
                        }
                        $value = [string]$literalMap[$qid][$litProp]
                        if ($value.Contains("T")) {
                            $value = $value.Split("T")[0]
                            if ($value.Length -ge 4) {
                                $value = $value.Substring(0, 4)
                            }
                        }
                        $props[$propName] = $value
                    }
                    if ($props.Count -gt 0) {
                        $literalRows.Add([pscustomobject]@{
                            title = $title
                            qid   = $qid
                            props = $props
                        })
                        $literalUpdatesThisBatch += $props.Count
                    }
                }

                foreach ($rel in $relationMap[$qid]) {
                    $mapping = $WikidataPropertyMap[$rel.prop]
                    if ($null -eq $mapping) {
                        continue
                    }

                    $targetName = if (-not [string]::IsNullOrWhiteSpace($rel.label)) { $rel.label } else { ($rel.value -split "/")[-1] }
                    if ([string]::IsNullOrWhiteSpace($targetName) -or $targetName -match "^Q\d+$") {
                        continue
                    }

                    $relationType = $mapping.relation
                    if (-not $relationRowsByType.ContainsKey($relationType)) {
                        $relationRowsByType[$relationType] = New-Object System.Collections.Generic.List[object]
                    }

                    $targetQid = $null
                    if ($rel.value -match "/entity/(Q\d+)$") {
                        $targetQid = $Matches[1]
                    }

                    $relationRowsByType[$relationType].Add([pscustomobject]@{
                        source_name = $title
                        source_qid  = $qid
                        target_name = $targetName
                        target_qid  = $targetQid
                        target_type = $mapping.target_type
                        property_id = $rel.prop
                    })
                    $relationsWrittenThisBatch += 1
                }
            }

            if (-not $DryRun) {
                if ($effectiveMode -eq "http") {
                    Write-Neo4jBatchHttp -BaseUri $baseUri -Username $resolvedNeo4jUsername -Password $resolvedNeo4jPassword -LiteralRows $literalRows -RelationRowsByType $relationRowsByType
                }
                else {
                    Write-Neo4jBatchCypherShell -CypherShellPath $CypherShellPath -Neo4jUri $resolvedNeo4jUri -Username $resolvedNeo4jUsername -Password $resolvedNeo4jPassword -LiteralRows $literalRows -RelationRowsByType $relationRowsByType
                }
            }

            $stats.qids_resolved += $qidsResolvedThisBatch
            $stats.literal_updates += $literalUpdatesThisBatch
            $stats.relations_written += $relationsWrittenThisBatch
            $stats.pages_processed += $batchTitles.Count
            $batchSucceeded = $true
            break
        }
        catch {
            if ($attempt -lt $MaxRetriesPerBatch) {
                Write-Warning ("Batch starting at index {0} failed on attempt {1}/{2}: {3}" -f ($effectiveStart + $offset), ($attempt + 1), ($MaxRetriesPerBatch + 1), $_.Exception.Message)
                Start-Sleep -Seconds ([Math]::Max(2, $PauseSeconds * ($attempt + 1)))
                continue
            }
            $stats.errors += 1
            $stats.pages_processed += $batchTitles.Count
            Write-Warning ("Batch starting at index {0} failed: {1}" -f ($effectiveStart + $offset), $_.Exception.Message)
        }
    }

    $batchIndex = [int]($offset / $BatchSize)
    if (($batchIndex % 5) -eq 0) {
        Write-Host ("Progress pages={0}/{1} qids={2} relations={3}" -f $stats.pages_processed, $stats.pages_total, $stats.qids_resolved, $stats.relations_written)
    }

    if ($PauseSeconds -gt 0) {
        Start-Sleep -Seconds $PauseSeconds
    }
}

$statsJson = $stats | ConvertTo-Json -Depth 10
Write-Host $statsJson
