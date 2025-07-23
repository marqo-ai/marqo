package ai.marqo.search;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.times;

import com.yahoo.component.chain.Chain;
import com.yahoo.document.*;
import com.yahoo.document.datatypes.Array;
import com.yahoo.document.datatypes.MapFieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.documentapi.*;
import com.yahoo.search.Query;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.SearchChainRegistry;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class PaginationExclusionsTest {

    private HybridSearcher hybridSearcher;
    private DocumentAccess documentAccess;
    private AsyncSession asyncSession;

    @BeforeEach
    void setUp() {
        documentAccess = mock(DocumentAccess.class);
        asyncSession = mock(AsyncSession.class);
        when(documentAccess.createAsyncSession(any())).thenReturn(asyncSession);
        hybridSearcher = new HybridSearcher(documentAccess);

        // Mock all constructions of DocumentUpdate
        DocumentTypeManager documentTypeManager = mock(DocumentTypeManager.class);
        when(documentAccess.getDocumentTypeManager()).thenReturn(documentTypeManager);
        DocumentType docType = mock(DocumentType.class);
        when(documentTypeManager.getDocumentType(anyString())).thenReturn(docType);

        // Mock the field and its data type hierarchy
        Field field = mock(Field.class);
        when(docType.getField("updated_at")).thenReturn(field);
        when(docType.getField(0)).thenReturn(field);
        when(field.getDataType()).thenReturn(DataType.LONG);
    }

    @Test
    void getPaginationDocIsNotTriggeredForOffset0() {
        Query query = createPaginationQuery(10, 0, "schema", "hash", 600, "disjunction");
        Chain<Searcher> searchChain = new Chain<>(hybridSearcher);
        Execution.Context context = Execution.Context.createContextStub((SearchChainRegistry) null);
        Execution execution = new Execution(searchChain, context);
        execution.search(query);
        verify(asyncSession, never()).get(any());
    }

    @Test
    void getPaginationDocIsTriggeredForOffsetGreaterThan0() {
        Query query = createPaginationQuery(10, 20, "schema", "hash", 600, "disjunction");
        Chain<Searcher> searchChain = new Chain<>(hybridSearcher);
        Execution.Context context = Execution.Context.createContextStub((SearchChainRegistry) null);
        Execution execution = new Execution(searchChain, context);
        execution.search(query);
        verify(asyncSession, times(1)).get(any());
    }

    @Test
    void getPaginationDocIsNotTriggeredForOffsetPlusLimitHigherThanPaginationCutoff()
            throws Exception {
        // Mock the response for getDocument
        Result result = mock(Result.class);
        when(result.isSuccess()).thenReturn(true);
        when(asyncSession.get(any())).thenReturn(result);

        DocumentResponse response = mock(DocumentResponse.class);
        when(asyncSession.getNext()).thenReturn(response);
        Document document = mock(Document.class);
        when(response.getDocument()).thenReturn(document);
        when(document.getFieldValue("offsets")).thenReturn(createOffsetMapFieldValue(10, 60));

        // offset + limit = 650 > pagination_limit_cutoff = 600, so should not do update
        Query query = createPaginationQuery(60, 600, "schema", "hash", 600, "disjunction");
        Chain<Searcher> searchChain = new Chain<>(hybridSearcher);
        Execution.Context context = Execution.Context.createContextStub((SearchChainRegistry) null);
        Execution execution = new Execution(searchChain, context);
        execution.search(query);

        // Verify that no update call was issued
        Thread.sleep(100);
        verify(asyncSession, never()).update(any());
    }

    @Test
    void updatePaginationDocIsNotTriggeredForJump() {
        // Simulate missing previous offset to indicate a jump
        Query query = createPaginationQuery(500, 200, "schema", "hash", 600, "disjunction");
        Chain<Searcher> searchChain = new Chain<>(hybridSearcher);
        Execution.Context context = Execution.Context.createContextStub((SearchChainRegistry) null);
        Execution execution = new Execution(searchChain, context);
        com.yahoo.search.Result result = execution.search(query);
        verify(asyncSession, never()).update(any());

        // Verify that the result contains metadata "no-cache" hit due to jump
    }

    @Test
    void updatePaginationDocIsTriggeredForOffset0() throws Exception {
        Query query = createPaginationQuery(10, 0, "schema", "hash", 600, "disjunction");
        Chain<Searcher> searchChain = new Chain<>(hybridSearcher);
        Execution.Context context = Execution.Context.createContextStub((SearchChainRegistry) null);
        Execution execution = new Execution(searchChain, context);
        execution.search(query);

        // Wait for the async operation to complete
        Thread.sleep(100);

        // Verify that an update call was issued with a DocumentUpdate mock
        verify(asyncSession, times(1)).update(any(DocumentUpdate.class));
    }

    @Test
    void updatePaginationDocIsTriggeredForNonJump() throws Exception {

        // Mock the response for getDocument
        Result result = mock(Result.class);
        when(result.isSuccess()).thenReturn(true);
        when(asyncSession.get(any())).thenReturn(result);

        DocumentResponse response = mock(DocumentResponse.class);
        when(asyncSession.getNext()).thenReturn(response);
        Document document = mock(Document.class);
        when(response.getDocument()).thenReturn(document);
        when(document.getFieldValue("offsets")).thenReturn(createOffsetMapFieldValue(2, 10));

        // Simulate existing previous offset (offset - limit = 10) to indicate a non-jump
        Query query = createPaginationQuery(10, 20, "schema", "hash", 600, "disjunction");
        Chain<Searcher> searchChain = new Chain<>(hybridSearcher);
        Execution.Context context = Execution.Context.createContextStub((SearchChainRegistry) null);
        Execution execution = new Execution(searchChain, context);
        execution.search(query);

        // Wait for the async operation to complete
        Thread.sleep(100);

        // Verify that an update call was issued with a DocumentUpdate mock
        verify(asyncSession, times(1)).update(any(DocumentUpdate.class));
    }

    // --- Helpers ---

    private Query createPaginationQuery(
            int limit, int offset, String schema, String hash, int cutoff, String retrievalMethod) {
        Query query = new Query("search/?query=test");
        query.properties().set("hits", limit);
        query.properties().set("offset", offset);
        query.properties().set("marqo__hybrid.pagination_schema", schema);
        query.properties().set("marqo__hybrid.pagination_hash", hash);
        query.properties().set("marqo__hybrid.pagination_limit_cutoff", cutoff);
        query.properties().set("marqo__hybrid.retrievalMethod", retrievalMethod);
        // Ensure valid rankingMethod: disjunction must use rrf
        String rankingMethodValue = "disjunction".equals(retrievalMethod) ? "rrf" : retrievalMethod;
        query.properties().set("marqo__hybrid.rankingMethod", rankingMethodValue);
        query.properties().set("marqo__yql.lexical", "lexical yql");
        query.properties().set("marqo__yql.tensor", "tensor yql");
        // Add required tensor rank features to avoid NullPointerException in createSubQuery
        TensorType tensorType = new TensorType.Builder().mapped("test_tensor").build();
        Tensor fieldsToRankLexical =
                Tensor.Builder.of(tensorType)
                        .cell(TensorAddress.ofLabels("marqo__lexical_text_field_1"), 1.0)
                        .cell(TensorAddress.ofLabels("marqo__lexical_text_field_2"), 1.0)
                        .build();
        Tensor fieldsToRankTensor =
                Tensor.Builder.of(tensorType)
                        .cell(TensorAddress.ofLabels("marqo__embeddings_text_field_1"), 1.0)
                        .cell(TensorAddress.ofLabels("marqo__embeddings_text_field_2"), 1.0)
                        .build();
        query.getRanking()
                .getFeatures()
                .put("query(marqo__fields_to_rank_lexical)", fieldsToRankLexical);
        query.getRanking()
                .getFeatures()
                .put("query(marqo__fields_to_rank_tensor)", fieldsToRankTensor);
        return query;
    }

    private MapFieldValue<StringFieldValue, Array<StringFieldValue>> createOffsetMapFieldValue(
            int offsets, int limit) {
        MapFieldValue<StringFieldValue, Array<StringFieldValue>> map =
                new MapFieldValue<>(
                        DataType.getMap(DataType.STRING, DataType.getArray(DataType.STRING)));
        for (int i = 0; i < offsets; i++) {
            StringFieldValue key = new StringFieldValue(String.valueOf(i * limit));
            Array<StringFieldValue> values = new Array<>(DataType.getArray(DataType.STRING));
            for (int j = 0; j < limit; j++) {
                values.add(new StringFieldValue("id" + (i * limit + j + 1)));
            }
            map.put(key, values);
        }
        return map;
    }
}
