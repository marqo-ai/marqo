package ai.marqo.index;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.yahoo.config.FileReference;
import java.io.IOException;
import java.util.Map;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class IndexSettingsTest {

    @Nested
    class TestLoadJsonStringToMap {

        @Test
        void shouldLoadObjectValues() throws IOException {
            Map<String, String> map = IndexSettings.parseJsonStringToMap("{\"a\": {\"aa\":\"b\"}}");
            assertThat(map.get("a")).isEqualTo("{\"aa\":\"b\"}");
        }

        @Test
        void shouldLoadArrayValues() throws IOException {
            Map<String, String> map =
                    IndexSettings.parseJsonStringToMap("{\"a\":[{\"aa\":\"b\"}]}");
            assertThat(map.get("a")).isEqualTo("[{\"aa\":\"b\"}]");
        }

        @Test
        void shouldLoadEmptyObject() throws IOException {
            Map<String, String> map = IndexSettings.parseJsonStringToMap("{}");
            assertThat(map).isEmpty();
        }

        @ParameterizedTest
        @ValueSource(
                strings = {
                    "{\"a\":\"b\"}",
                    "{\"a\":1}",
                    "{\"a\":2.2}",
                    "{\"a\":null}",
                    "{\"a\":true}",
                    "[]", // array is not a valid json object
                    "", // empty string is not a valid json object
                })
        void shouldThrowIllegalArgumentExceptionForOtherTypes(String jsonString) {
            assertThatThrownBy(() -> IndexSettings.parseJsonStringToMap(jsonString))
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }

    @Nested
    class TestLoadIndexSettingFiles {

        @Test
        void shouldReturnEmptyIndexSettingsIfNoFileExists() {
            IndexSettings indexSettings =
                    new IndexSettings(
                            new ai.marqo.index.IndexSettingsConfig.Builder()
                                    .indexSettingsFile(new FileReference("does_not_exist.json"))
                                    .indexSettingsHistoryFile(
                                            new FileReference("does_not_exist.json"))
                                    .build());
            assertThat(indexSettings.getAllIndexSettings()).isEqualTo("[]");
            assertThat(indexSettings.getAllIndexSettingsHistory()).isEqualTo("[]");
        }

        @Test
        void shouldReturnEmptyIndexSettingsFromEmptyFiles() {
            IndexSettings indexSettings =
                    new IndexSettings(
                            new ai.marqo.index.IndexSettingsConfig.Builder()
                                    .indexSettingsFile(
                                            new FileReference(
                                                    "src/test/resources/index-settings/index_settings_empty.json"))
                                    .indexSettingsHistoryFile(
                                            new FileReference(
                                                    "src/test/resources/index-settings/index_settings_history_empty.json"))
                                    .build());
            assertThat(indexSettings.getAllIndexSettings()).isEqualTo("[]");
            assertThat(indexSettings.getAllIndexSettingsHistory()).isEqualTo("[]");
        }

        @Test
        void shouldReturnIndexSettingsFromFiles() {
            IndexSettings indexSettings =
                    new IndexSettings(
                            new ai.marqo.index.IndexSettingsConfig.Builder()
                                    .indexSettingsFile(
                                            new FileReference(
                                                    "src/test/resources/index-settings/index_settings.json"))
                                    .indexSettingsHistoryFile(
                                            new FileReference(
                                                    "src/test/resources/index-settings/index_settings_history.json"))
                                    .build());

            assertThat(indexSettings.getAllIndexSettings())
                    .isEqualTo("[{\"name\":\"index1\",\"version\":2}]");
            assertThat(indexSettings.getIndexSetting("index1"))
                    .isEqualTo("{\"name\":\"index1\",\"version\":2}");
            assertThat(indexSettings.getIndexSettingHistory("index1"))
                    .isEqualTo("[{\"name\":\"index1\",\"version\":1}]");
            assertThat(indexSettings.getIndexSetting("index2")).isNull();
            assertThat(indexSettings.getIndexSettingHistory("index2")).isNull();
        }
    }
}
