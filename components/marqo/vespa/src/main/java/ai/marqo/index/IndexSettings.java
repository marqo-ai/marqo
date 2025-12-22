package ai.marqo.index;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.common.annotations.VisibleForTesting;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IndexSettings {
    private static final Logger logger = LoggerFactory.getLogger(IndexSettings.class);
    private static final ObjectMapper mapper = new ObjectMapper();

    private final Map<String, String> indexSettings;
    private final Map<String, String> indexSettingsHistory;
    private final String allIndexSettings;
    private final String allIndexSettingsHistory;

    public IndexSettings(ai.marqo.index.IndexSettingsConfig config) {
        indexSettings = loadIndexSettingsFromFile(config.indexSettingsFile());
        indexSettingsHistory = loadIndexSettingsFromFile(config.indexSettingsHistoryFile());
        allIndexSettings = toJsonArray(indexSettings);
        allIndexSettingsHistory = toJsonArray(indexSettingsHistory);
    }

    private String toJsonArray(Map<String, String> jsonMap) {
        return jsonMap.values().stream().collect(Collectors.joining(",", "[", "]"));
    }

    private Map<String, String> loadIndexSettingsFromFile(Path path) {
        try {
            String content = Files.readString(path);
            return parseJsonStringToMap(content);
        } catch (NoSuchFileException e) {
            logger.warn("File not found at {}", path);
            return new HashMap<>();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load index settings: " + path, e);
        }
    }

    @VisibleForTesting
    static Map<String, String> parseJsonStringToMap(String jsonString) throws IOException {
        JsonNode jsonNode = mapper.readTree(jsonString);

        if (!(jsonNode instanceof ObjectNode)) {
            throw new IllegalArgumentException("Invalid input. Expected a JSON object.");
        }

        Map<String, String> map = new HashMap<>();

        jsonNode.fields()
                .forEachRemaining(
                        field -> {
                            JsonNode value = field.getValue();
                            if (value instanceof ObjectNode || value instanceof ArrayNode) {
                                map.put(field.getKey(), field.getValue().toString());
                            } else {
                                throw new IllegalArgumentException(
                                        "Invalid Json object or array: " + field.getValue());
                            }
                        });

        return map;
    }

    public String getIndexSetting(String name) {
        return indexSettings.get(name);
    }

    public String getAllIndexSettings() {
        return allIndexSettings;
    }

    public String getIndexSettingHistory(String name) {
        return indexSettingsHistory.get(name);
    }

    public String getAllIndexSettingsHistory() {
        return allIndexSettingsHistory;
    }
}
