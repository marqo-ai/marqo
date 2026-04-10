package ai.marqo.index;

import static com.yahoo.jdisc.http.HttpRequest.Method.GET;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.container.jdisc.HttpRequest;
import com.yahoo.container.jdisc.HttpResponse;
import com.yahoo.container.jdisc.ThreadedHttpRequestHandler;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executor;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IndexSettingRequestHandler extends ThreadedHttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(IndexSettingRequestHandler.class);

    private static final Pattern INDEX_SETTINGS_PATH_PATTERN =
            Pattern.compile("/index-settings/(.+)");

    private final IndexSettings indexSettings;

    public IndexSettingRequestHandler(Executor executor, IndexSettings indexSettings) {
        super(executor);
        this.indexSettings = indexSettings;
    }

    @Override
    public HttpResponse handle(HttpRequest httpRequest) {
        if (!httpRequest.getMethod().equals(GET)) {
            return JsonResponse.error(405, "Only GET requests are allowed");
        }

        String path = httpRequest.getUri().getPath();
        if (path.equals("/index-settings")) {
            return JsonResponse.success(indexSettings.getAllIndexSettings());
        } else {
            Matcher matcher = INDEX_SETTINGS_PATH_PATTERN.matcher(path);
            if (!matcher.find()) {
                return JsonResponse.error(400, String.format("Uri path '%s' is invalid", path));
            }

            // TODO handle URL decoding
            String indexName = matcher.group(1);
            try {
                String indexSetting = indexSettings.getIndexSetting(indexName);
                if (indexSetting == null) {
                    return JsonResponse.error(
                            404, String.format("Index setting '%s' does not exist", indexName));
                }
                return JsonResponse.success(indexSetting);
            } catch (Exception e) {
                logger.error("Failed to get index setting: {}", indexName, e);
                return JsonResponse.error(
                        500, "Failed to get index setting: " + indexName + ": " + e.getMessage());
            }
        }
    }

    static class JsonResponse extends HttpResponse {
        private static final ObjectMapper mapper = new ObjectMapper();
        private final byte[] data;

        private JsonResponse(int code, String data) {
            super(code);
            this.data = data.getBytes(StandardCharsets.UTF_8);
        }

        public static JsonResponse error(int code, String message) {
            try {
                return new JsonResponse(code, mapper.writeValueAsString(new ErrorMessage(message)));
            } catch (JsonProcessingException e) {
                return new JsonResponse(code, "");
            }
        }

        public static JsonResponse success(String data) {
            return new JsonResponse(200, data);
        }

        public String getContentType() {
            return "application/json";
        }

        public void render(OutputStream outputStream) throws IOException {
            outputStream.write(this.data);
        }
    }

    record ErrorMessage(String error) {}
}
